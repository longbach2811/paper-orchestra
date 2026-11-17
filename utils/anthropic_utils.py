# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Anthropic (Claude) backend, added to let PaperOrchestra use Claude models
alongside the built-in Gemini/OpenAI backends. Mirrors the interface of
openai_utils.py so it plugs into utils/llm_backend_utils.py the same way.
"""

from typing import Any
import re
import json
import time
import os
import threading
import base64
import mimetypes

from anthropic import Anthropic  # type: ignore
from dotenv import load_dotenv  # type: ignore

dot_file = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(dot_file)

_anthropic_client = None
_anthropic_client_lock = threading.Lock()

DEFAULT_MAX_TOKENS = 8192


def get_anthropic_client() -> Anthropic:
    """Returns a shared Anthropic client, creating it on first use.

    The client is created lazily so that ANTHROPIC_API_KEY is only required
    when a Claude model is actually called (e.g. Gemini/OpenAI-only setups
    don't need it).
    """
    global _anthropic_client
    if _anthropic_client is None:
        with _anthropic_client_lock:
            if _anthropic_client is None:
                anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not anthropic_api_key:
                    raise ValueError(
                        "ANTHROPIC_API_KEY must be set to use Claude models."
                    )
                _anthropic_client = Anthropic(api_key=anthropic_api_key)
    return _anthropic_client


def parse_anthropic_json_results(response: str):
    if not response or not isinstance(response, str):
        return None

    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, response, re.DOTALL)

    if not matches:
        json_pattern = r"(\{.*\}|\[.*\])"
        matches = re.findall(json_pattern, response, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            try:
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                json_string_clean = re.sub(r",\s*([\]}])", r"\1", json_string_clean)
                json_string_clean = re.sub(
                    r'\\(?![\\"/bfnrtu])', r"\\\\", json_string_clean
                )
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue

    try:
        start_idx = response.find("{")
        end_idx = response.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_string = response[start_idx : end_idx + 1]
            json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
            json_string_clean = re.sub(r",\s*([\]}])", r"\1", json_string_clean)
            json_string_clean = re.sub(
                r'\\(?![\\"/bfnrtu])', r"\\\\", json_string_clean
            )
            return json.loads(json_string_clean)
    except Exception:
        pass

    try:
        start_idx = response.find("[")
        end_idx = response.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_string = response[start_idx : end_idx + 1]
            json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
            json_string_clean = re.sub(r",\s*([\]}])", r"\1", json_string_clean)
            json_string_clean = re.sub(
                r'\\(?![\\"/bfnrtu])', r"\\\\", json_string_clean
            )
            return json.loads(json_string_clean)
    except Exception:
        pass

    return None  # No valid JSON found


def _extract_text(response) -> str:
    """Concatenates all text blocks from an Anthropic Messages response."""
    text_parts = [block.text for block in response.content if block.type == "text"]
    return "\n".join(text_parts)


def call_anthropic_models_with_content(
    content: list[Any],
    model_name: str,
    max_retries: int = 5,
    base_interval_sec: int = 5,
    result_parsing_func: callable = parse_anthropic_json_results,
    generation_configs: dict = {},
    check_parsed_response_not_none: bool = True,
    system_prompt: str = None,
):
    raw_response = None
    parsed_response = None
    anthropic_client = get_anthropic_client()

    configs = dict(generation_configs) if generation_configs else {}
    max_tokens = configs.pop("max_tokens", DEFAULT_MAX_TOKENS)

    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model_name,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": content}],
                **configs,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = anthropic_client.messages.create(**kwargs)
            raw_response = _extract_text(response)
            parsed_response = result_parsing_func(raw_response)

            if check_parsed_response_not_none:
                assert (
                    parsed_response is not None
                ), f"Parsed response is None, Here is the raw response:\n{raw_response}"

            return {
                "raw_response": raw_response,
                "parsed_response": parsed_response,
            }

        except Exception as e:
            print(
                f"Warning: Anthropic API call (model={model_name}) failed on attempt {attempt + 1} of {max_retries}. Error: {e}",
            )

            if attempt + 1 == max_retries:
                print(f"Error: All {max_retries} retry attempts failed.")
                raise e

            delay = base_interval_sec * (2 ** (attempt - 1)) if attempt > 0 else 0
            print(f"Retrying in {delay} seconds...")
            if delay > 0:
                time.sleep(delay)

    print("Error: Exited retry loop unexpectedly.")
    return {
        "raw_response": raw_response,
        "parsed_response": parsed_response,
    }


def call_anthropic_models_with_text_prompt(
    prompt: str,
    model_name: str,
    result_parsing_func: callable = parse_anthropic_json_results,
    generation_configs: dict = {},
    check_parsed_response_not_none: bool = True,
    max_retries: int = 5,
    base_interval_sec: int = 5,
    system_prompt: str = None,
):
    return call_anthropic_models_with_content(
        content=[{"type": "text", "text": prompt}],
        model_name=model_name,
        result_parsing_func=result_parsing_func,
        generation_configs=generation_configs,
        check_parsed_response_not_none=check_parsed_response_not_none,
        max_retries=max_retries,
        base_interval_sec=base_interval_sec,
        system_prompt=system_prompt,
    )


def get_anthropic_image_part_from_path(image_path: str):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    media_type = mimetypes.guess_type(image_path)[0] or "image/png"

    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64_image,
            },
        }
    ]


def get_anthropic_pdf_part_from_path(pdf_path: str):
    with open(pdf_path, "rb") as pdf_file:
        base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")

    return [
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64_pdf,
            },
        }
    ]

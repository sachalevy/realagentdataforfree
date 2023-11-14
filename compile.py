from pathlib import Path
import datetime
import os
import json
import re
import base64

import requests
import cv2
import pytesseract
from dotenv import load_dotenv

load_dotenv(".env")


headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
}

SYSTEM_PROMPT = """You are a helpful assistant and expert in APIs.
Your task is to represent the action being performed by a computer user from a screenshot into a goal-oriented API call.
For example if a user is navigating to a weather website, you should return an API call representing the action of getting the weather.
In addition to the screenshot, you will be provided with a textual description of the user's current action (ACTION_DESCRIPTION),
the user's recent keyboard entries (USER_INPUT), text extracted from the user's screen (WINDOW_CONTEXT), and the name of the active app on the user's computer (ACTIVE_APP).
Only respond with the JSON formatted API call.
Synthesise the information in an API endpoint signature in the following format:
```
{
  "name": "get_weather",
  "description": "Determine weather in my location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "The city and state e.g. San Francisco, CA"
      },
      "unit": {
        "type": "string",
        "enum": [
          "c",
          "f"
        ]
      }
    },
    "required": [
      "location"
    ]
  }
}
```
"""


def get_api_signature_from_vision(
    filepath: Path,
    action_description: str,
    sentences: str,
    text_context: str,
    active_app: str,
):
    image_payload = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(filepath)}"},
    }
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Provide the corresponding JSON signature for the API endpoint representing the user's action.\
                            \nACTION_DESCRIPTION:```{action_description}```\nUSER_INPUT:```{user_input}```\nWINDOW_CONTEXT:```{window_context}```\nACTIVE_APP:```{active_app}```".format(
                            action_description=action_description,
                            user_input=" ".join(sentences),
                            window_context=text_context,
                            active_app=active_app,
                        ),
                    },
                    image_payload,
                ],
            },
        ],
        "max_tokens": 512,
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
    except Exception as e:
        return {}, None

    try:
        loaded_resp = response.json()
    except Exception as e:
        return {}, None

    try:
        content = response.json().get("choices")[0].get("message").get("content")
        return loaded_resp, content
    except Exception as e:
        return loaded_resp, None


SYSTEM_PROMPT_API_SIGNATURE_FROM_TEXT = """You are a helpful assistant and expert in APIs.
Your task is to represent the action being performed by a computer user from text extracted from a screenshot of the user's screen into a goal-oriented API endpoint signature.
For example if a user is navigating to a weather website, you should return an API call representing the action of getting the weather.
In addition to the screenshot, you will be provided with a textual description of the user's current action (ACTION_DESCRIPTION),
the user's recent keyboard entries (USER_INPUT), text extracted from the user's screen (WINDOW_CONTEXT), and the name of the active app on the user's computer (ACTIVE_APP).
Only respond with the JSON formatted API call.
Synthesise the information in an API endpoint signature in the following format:
```
{
  "name": "get_weather",
  "description": "Determine weather in my location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "The city and state e.g. San Francisco, CA"
      },
      "unit": {
        "type": "string",
        "enum": [
          "c",
          "f"
        ]
      }
    },
    "required": [
      "location"
    ]
  }
}
```
"""


def get_api_signature_from_text(
    action_description: str,
    sentences: str,
    text_context: str,
    active_app: str,
):
    payload = {
        "model": "gpt-3.5-turbo-1106",
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_API_SIGNATURE_FROM_TEXT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Provide the corresponding JSON signature for the API endpoint representing the user's action.\
                            \nACTION_DESCRIPTION:```{action_description}```\nUSER_INPUT:```{user_input}```\nWINDOW_CONTEXT:```{window_context}```\nACTIVE_APP:```{active_app}```".format(
                            action_description=action_description,
                            user_input=" ".join(sentences),
                            window_context=text_context,
                            active_app=active_app,
                        ),
                    },
                ],
            },
        ],
        "max_tokens": 512,
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
    except Exception as e:
        return {}, None

    try:
        loaded_resp = response.json()
    except Exception as e:
        return {}, None

    try:
        content = response.json().get("choices")[0].get("message").get("content")
        return loaded_resp, content
    except Exception as e:
        return loaded_resp, None


TEXT_SYSTEM_PROMPT = """
You are a helpful assistant and expert at labelling user actions in human-computer interactions.
You are given the current active app, the text extracted from a screenshot of the user's screen,
and all characters entered by the user during a short time interval around the time of screenshot.
Indicate the user's action/intent based on the text displayed on their screen and the characters they entered.
The goal is to contextualize the current task within the user's workflow.
"""


def get_user_action_description(sentences, text_context, active_app):
    payload = {
        "model": "gpt-3.5-turbo-1106",
        "messages": [
            {"role": "system", "content": TEXT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the user's action based on content displayed on their screen.\
                            The current active app on the user's computer is {active_app}.\
                            \nDISPLAYED_TEXT:```{displayed_text}```\nUSER_INPUT:```{user_input}```".format(
                            displayed_text=text_context,
                            user_input=" ".join(sentences),
                            active_app=active_app,
                        ),
                    },
                ],
            },
        ],
        "max_tokens": 512,
    }
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
        loaded_resp, content = response.json(), response.json().get("choices")[0].get(
            "message"
        ).get("content")
        return loaded_resp, content
    except Exception as e:
        return {}, None


def mock_get_api_signature(filepath):
    api_signature = {
        "name": "create_presentation_slide",
        "description": "Generate a presentation slide with iterative spatial integration visualizations",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Title for the slide"},
                "iterations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "iteration_number": {
                                "type": "integer",
                                "description": "The iteration step of the spatial integration process",
                            },
                            "image": {
                                "type": "string",
                                "description": "Base64 encoded image of the spatial integration visualization",
                            },
                        },
                        "required": ["iteration_number", "image"],
                    },
                },
            },
            "required": ["title", "iterations"],
        },
    }

    return {}, api_signature


def retrieve_current_event(filepath: Path, start_ts: int, end_ts: int):
    events = []
    with open(filepath, "r") as file:
        for line in file:
            parts = line.strip().split(",")
            ts = float(parts[-1]) if filepath.stem != "clicks" else float(parts[-2])
            if ts >= start_ts and ts <= end_ts:
                events.append(line.strip())

    if filepath.stem == "clicks":
        return events, events[0].strip().split(",")[-1]

    return events


def extract_sentences_from_keystrokes(keystrokes):
    sentences = []
    current_sentence = ""
    shift_pressed = False

    for line in keystrokes:
        parts = line.strip().split(",")
        key, action = parts[0], parts[1]

        if key == "Key.shift":
            if action == "press":
                shift_pressed = True
            elif action == "release":
                shift_pressed = False
        elif action == "press":
            if key.startswith("'") and key.endswith("'"):
                char = key[1:-1]
                if shift_pressed:
                    char = char.upper()
                current_sentence += char
            elif key == "Key.space":
                current_sentence += " "
            elif key == "Key.enter":
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            elif key == "Key.backspace":
                current_sentence = current_sentence[:-1]

    if current_sentence:
        sentences.append(current_sentence.strip())

    return sentences


def clean_ocr_text(text):
    text = re.sub(r"\s+", " ", text.strip())
    text = text.replace("\n\n", "\n")
    text = text.encode("ascii", "ignore").decode()

    return text


def extract_image_text(image_path):
    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_string(255 - gray)
    return clean_ocr_text(data)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


SAMPLE_PROMPT = "SYSTEM: You are an helpful assistant who has access to the following functions to help the user, you can use the functions if needed- {api_signature}\
    USER: {inferred_user_prompt}\
    ASSISTANT: <functioncall> {inferred_function_arguments} FUNCTION RESPONSE: {inferred_function_response}\
    ASSISTANT: {assistant_response}"


def format_sample(
    api_signature,
    inferred_user_prompt,
    inferred_function_arguments,
    inferred_function_response,
    inferred_assistant_response,
):
    return SAMPLE_PROMPT.format(
        api_signature=api_signature,
        inferred_user_prompt=inferred_user_prompt,
        inferred_function_arguments=inferred_function_arguments,
        inferred_function_response=inferred_function_response,
        assistant_response=inferred_assistant_response,
    )


SAMPLE_MD_PROMPT = "SYSTEM: You are an helpful assistant who has access to the following functions to help the user, you can use the functions if needed- ```json\n{api_signature}\n```\
    \nUSER: {inferred_user_prompt}\
    \nASSISTANT: <functioncall> \n```json\n{inferred_function_arguments}\n```\n\
    \nFUNCTION RESPONSE: \n```json\n{inferred_function_response}\n```\n\
    \nASSISTANT: {assistant_response}"


def format_sample_md(
    api_signature,
    inferred_user_prompt,
    inferred_function_arguments,
    inferred_function_response,
    inferred_assistant_response,
):
    return SAMPLE_MD_PROMPT.format(
        api_signature=api_signature,
        inferred_user_prompt=inferred_user_prompt,
        inferred_function_arguments=inferred_function_arguments,
        inferred_function_response=inferred_function_response,
        assistant_response=inferred_assistant_response,
    )


INFER_ARGS_SYSTEM_PROMPT = """
You are a helpful assistant and expert in APIs. You are to help \
    in formatting the intent of a user's action into a textual conversation centered around \
    the usage of an API tool. You are given text extracted from a screenshot of a user's \
    computer screen as well as the name of the active app on the user's computer. Recent keystrokes \
    of the user are also provided. In addition, you are provided with the API signature representing \
    the current action of the user, as well as a description of their action.

Expect an input formatted as:
``
ACTION_DESCRIPTION:```{action_description}```
USER_INPUT:```{user_input}```
WINDOW_CONTEXT:```{window_context}```
ACTIVE_APP:```{active_app}```
API_SIGNATURE:```{api_signature}```
``

Your task is to use these parameter to infer the {{user_prompt}}, {{function_arguments}}, \
    {{function_response}}, and {{assistant_response}} arguments in the following template:
``
SYSTEM: You are an helpful assistant who has access to the following functions to help the user, you can use the functions if needed- {{api_signature}}
USER: {{user_prompt}}
ASSISTANT: <functioncall> {{function_arguments}} FUNCTION RESPONSE: {{function_response}}
ASSISTANT: {{assistant_response}}
``

Make sure your response is formatted as a JSON object with the following keys:
```
{
    "user_prompt": "The prompt/question the user is asking when completing this action.",
    "function_arguments": "The arguments to required to call the specific function.",
    "function_response": "The response returned by the function call (ie result of the user's action).",
    "assistant_response": "The outcome of the user action, phrased as if the assistant is responding to the user."
}
```
"""

INFER_ARGS_USER_PROMPT = """
Specify the values for the following keys:
``
{{
    "user_prompt": "The prompt/question the user is asking when completing their action (eg if writing code this would be `implement function X`).",
    "function_arguments": "The arguments required to call the specific function (eg if searching information online about someone, their name would be required).",
    "function_response": "The response returned by the function call (ie result of the user's action).",
    "assistant_response": "The outcome of the user action, phrased as if the assistant is responding to the user."
}}
``

Use the following parameters to infer the values for the keys:
``
ACTION_DESCRIPTION:```{action_description}```
USER_INPUT:```{user_input}```
WINDOW_CONTEXT:```{window_context}```
ACTIVE_APP:```{active_app}```
API_SIGNATURE:```{api_signature}```
``

Make sure to format your answer as a JSON object.
"""


def infer_args(action_description, sentences, text_context, active_app, api_signature):
    formatted_prompt = INFER_ARGS_USER_PROMPT.format(
        action_description=action_description,
        user_input=" ".join(sentences),
        window_context=text_context,
        active_app=active_app,
        api_signature=api_signature,
    )
    payload = {
        "model": "gpt-3.5-turbo-1106",
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": INFER_ARGS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": formatted_prompt},
                ],
            },
        ],
        "max_tokens": 512,
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
    except Exception as e:
        return {}, None

    print("inferred", response.json())

    try:
        loaded_resp = response.json()
    except Exception as e:
        return {}, None

    to_json = lambda x: json.loads(x) if isinstance(x, str) else x
    try:
        content = response.json().get("choices")[0].get("message").get("content")
        print(content, type(content))
        content = to_json(content)
        return loaded_resp, content
    except Exception as e:
        return loaded_resp, None


def main(use_vision=False):
    unique_screenshot_dir = Path(
        "/Users/sachalevy/IMPLEMENT/datamakr/data/unique_screenshots"
    )
    screenshot_filenames = [x.stem for x in unique_screenshot_dir.iterdir()]
    screenshot_filenames = sorted(
        screenshot_filenames, key=lambda x: int(x.split("_")[1])
    )
    screenshot_timestamps = [int(x.split("_")[1]) for x in screenshot_filenames]

    keystroke_filepath = Path("/Users/sachalevy/IMPLEMENT/datamakr/data/presses.txt")
    scrolls_filepath = Path("/Users/sachalevy/IMPLEMENT/datamakr/data/scrolls.txt")
    clicks_filepath = Path("/Users/sachalevy/IMPLEMENT/datamakr/data/clicks.txt")

    realtime_output = Path("/Users/sachalevy/IMPLEMENT/datamakr/data/realtime.md")
    realtime_fd = open(realtime_output, "a")

    # iterate through all files
    examples, samples = [], []
    for i in range(3, 10):
        screenshot_filename = screenshot_filenames[i]
        start_ts, end_ts = screenshot_timestamps[i], screenshot_timestamps[i + 1]
        print(
            "screenshot",
            screenshot_filename,
            i,
            datetime.datetime.fromtimestamp(start_ts),
            datetime.datetime.fromtimestamp(end_ts),
        )
        keystrokes = retrieve_current_event(keystroke_filepath, start_ts, end_ts)
        scrolls = retrieve_current_event(scrolls_filepath, start_ts, end_ts)
        clicks, active_app = retrieve_current_event(clicks_filepath, start_ts, end_ts)

        # assemble all current keystrokes
        sentences = extract_sentences_from_keystrokes(keystrokes)
        print(
            "user actions", len(sentences), len(keystrokes), len(scrolls), len(clicks)
        )

        # extract text from screenshot
        screenshot_filepath = unique_screenshot_dir / (screenshot_filename + ".png")
        text_context = extract_image_text(screenshot_filepath)
        print("window context", text_context)

        full_text_response, action_description = get_user_action_description(
            sentences, text_context, active_app
        )
        print(action_description)

        # API call from GPT-4-vision
        if use_vision:
            full_vision_response, api_signature = get_api_signature_from_vision(
                screenshot_filepath,
                action_description,
                sentences,
                text_context,
                active_app,
            )
        else:
            full_vision_response, api_signature = get_api_signature_from_text(
                action_description,
                sentences,
                text_context,
                active_app,
            )
        print(json.dumps(full_vision_response))
        print("api signature", api_signature)

        arg_inference_response, inferred_args = infer_args(
            action_description, sentences, text_context, active_app, api_signature
        )
        user_prompt = inferred_args.get("user_prompt")
        function_arguments = inferred_args.get("function_arguments")
        function_response = inferred_args.get("function_response")
        assistant_response = inferred_args.get("assistant_response")

        sample = format_sample(
            api_signature,
            user_prompt,
            function_arguments,
            function_response,
            assistant_response,
        )
        samples.append(sample)

        # print(sample)
        #
        # md_sample = format_sample_md(
        #    api_signature,
        #    user_prompt,
        #    function_arguments,
        #    function_response,
        #    assistant_response,
        # )
        realtime_fd.write(sample + "\n")

        example = {
            "screenshot_filepath": str(screenshot_filepath),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration": end_ts - start_ts,
            "active_app": active_app,
            "action_description": action_description,
            "api_signature": json.dumps(api_signature)
            if isinstance(api_signature, dict)
            else api_signature,
            "user_input": " ".join(sentences),
            "metadata": {
                "full_text_response": full_text_response,
                "full_vision_response": full_vision_response,
                "keystrokes": keystrokes,
                "scrolls": scrolls,
                "clicks": clicks,
            },
        }
        examples.append(example)

        # make an .md sample and add it live to a file

        exit(0)

    output_examples_filepath = Path("data/examples.json")
    with open(output_examples_filepath, "w") as file:
        json.dump(examples, file)


if __name__ == "__main__":
    main()

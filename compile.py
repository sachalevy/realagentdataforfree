from pathlib import Path
import os
import json

import requests
from dotenv import load_dotenv

import prompt, utils

load_dotenv(".env")


headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
}


def get_api_signature_from_vision(
    filepath: Path,
    action_description: str,
    sentences: str,
    text_context: str,
    active_app: str,
):
    image_payload = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{utils.encode_image(filepath)}"},
    }
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "system",
                "content": prompt.SYSTEM_PROMPT_API_SIGNATURE_FROM_VISION,
            },
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
            {"role": "system", "content": prompt.SYSTEM_PROMPT_API_SIGNATURE_FROM_TEXT},
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


def get_user_action_description(sentences, text_context, active_app):
    payload = {
        "model": "gpt-3.5-turbo-1106",
        "messages": [
            {"role": "system", "content": prompt.TEXT_SYSTEM_PROMPT},
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


def infer_args(action_description, sentences, text_context, active_app, api_signature):
    formatted_prompt = prompt.INFER_ARGS_USER_PROMPT.format(
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
            {"role": "system", "content": prompt.INFER_ARGS_SYSTEM_PROMPT},
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
    unique_screenshot_dir = Path("data/unique_screenshots")
    screenshot_filenames = [x.stem for x in unique_screenshot_dir.iterdir()]
    screenshot_filenames = sorted(
        screenshot_filenames, key=lambda x: int(x.split("_")[1])
    )
    screenshot_timestamps = [int(x.split("_")[1]) for x in screenshot_filenames]

    keystroke_filepath = Path("data/presses.txt")
    scrolls_filepath = Path("data/scrolls.txt")
    clicks_filepath = Path("data/clicks.txt")

    realtime_output = Path("data/realtime.txt")
    realtime_fd = open(realtime_output, "a")

    # iterate through all files
    complete_samples, samples = [], []
    for i in range(len(screenshot_filenames) - 1):
        screenshot_filename = screenshot_filenames[i]
        start_ts, end_ts = screenshot_timestamps[i], screenshot_timestamps[i + 1]

        keystrokes = utils.retrieve_current_event(keystroke_filepath, start_ts, end_ts)
        scrolls = utils.retrieve_current_event(scrolls_filepath, start_ts, end_ts)
        clicks, active_app = utils.retrieve_current_event(
            clicks_filepath, start_ts, end_ts
        )

        sentences = utils.extract_sentences_from_keystrokes(keystrokes)
        screenshot_filepath = unique_screenshot_dir / (screenshot_filename + ".png")
        text_context = utils.extract_image_text(screenshot_filepath)

        full_text_response, action_description = get_user_action_description(
            sentences, text_context, active_app
        )

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

        _, inferred_args = infer_args(
            action_description, sentences, text_context, active_app, api_signature
        )

        sample = prompt.SAMPLE_PROMPT.format(
            api_signature=api_signature,
            inferred_user_prompt=inferred_args.get("user_prompt"),
            inferred_function_arguments=inferred_args.get("function_arguments"),
            inferred_function_response=inferred_args.get("function_response"),
            assistant_response=inferred_args.get("assistant_response"),
        )
        samples.append(sample)
        realtime_fd.write(sample + "\n")

        complete_sample = {
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
        complete_samples.append(complete_sample)

    output_complete_samples = Path("data/complete_samples.json")
    with open(output_complete_samples, "w") as file:
        json.dump(complete_samples, file)


if __name__ == "__main__":
    main()

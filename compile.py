from pathlib import Path
import os
import json
import argparse

import openai
import requests
from dotenv import load_dotenv

import prompt, utils

load_dotenv(".env")

client = openai.OpenAI()
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
}


def get_api_signature_from_vision(filepath: Path, user_prompt: str):
    """
    Ask gpt-4v to generate an api signature describing the user's actions
    based on the provided screenshot.
    """
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
                    {"type": "text", "text": user_prompt},
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
        return response.json(), response.json().get("choices")[0].get("message").get(
            "content"
        )
    except Exception as e:
        return {}, None


def get_completion(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-3.5-turbo-1106",
    request_json: bool = False,
):
    completion_kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    if request_json:
        completion_kwargs["response_format"] = {"type": "json_object"}
    completion = client.chat.completions.create(**completion_kwargs)

    return completion, completion.choices[0].message.content


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

        # get all events happening between start_ts and end_ts
        keystrokes = utils.retrieve_current_event(keystroke_filepath, start_ts, end_ts)
        scrolls = utils.retrieve_current_event(scrolls_filepath, start_ts, end_ts)
        clicks, active_app = utils.retrieve_current_event(
            clicks_filepath, start_ts, end_ts
        )

        # extract user entries + on-screen text context
        sentences = utils.extract_sentences_from_keystrokes(keystrokes)
        screenshot_filepath = unique_screenshot_dir / (screenshot_filename + ".png")
        text_context = utils.extract_text_from_screenshot(screenshot_filepath)
        action_description_user_prompt = prompt.USER_PROMPT_ACTION_DESCRIPTION.format(
            displayed_text=text_context,
            user_input=" ".join(sentences),
            active_app=active_app,
        )
        raw_action_description_completion, action_description = get_completion(
            prompt.SYSTEM_PROMPT_ACTION_DESCRIPTION,
            action_description_user_prompt,
            request_json=True,
        )
        print(action_description)

        api_signature_user_prompt = prompt.USER_PROMPT_API_SIGNATURE.format(
            action_description=action_description,
            user_input=" ".join(sentences),
            window_context=text_context,
            active_app=active_app,
        )
        if use_vision:
            raw_api_signature_completion, api_signature = get_api_signature_from_vision(
                screenshot_filepath, api_signature_user_prompt
            )
        else:
            raw_api_signature_completion, api_signature = get_completion(
                prompt.SYSTEM_PROMPT_API_SIGNATURE_FROM_TEXT, api_signature_user_prompt
            )
        print(api_signature)

        inferred_api_call_arguments_user_prompt = prompt.INFER_ARGS_USER_PROMPT.format(
            action_description=action_description,
            user_input=" ".join(sentences),
            window_context=text_context,
            active_app=active_app,
            api_signature=api_signature,
        )
        print(inferred_api_call_arguments_user_prompt)
        raw_arg_completion, inferred_args = get_completion(
            prompt.INFER_ARGS_SYSTEM_PROMPT, inferred_api_call_arguments_user_prompt
        )
        print(inferred_args)

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
                "raw_action_description_completion": raw_action_description_completion,
                "raw_api_signature_completion": raw_api_signature_completion,
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
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--use-vision",
        action="store_true",
        help="Use vision model to extract api signature",
    )
    main()

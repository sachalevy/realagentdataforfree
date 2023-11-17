# datamakr
Compile human-computer interaction datasets & convert them to text.

The goal is to facilitate the extraction of high-quality agent training data from humans performing tasks on their computers.

For example, think of the steps a data scientists goes through to answer a question about an ongoing training run. They receive a slack message (*1*), navigate to wandb on their browser (*2*), search for the training run in question (*3*), extract the relevant information (*4*), and answer their colleague on slack (*5*).

This can easily be described through text, given the necessary information (like screenshots and user activity recording). Inspired by OpenAI's function calls (for machines), we try to transcribe these workflows in an API-like format. For example for step 3:
```markdown
SYSTEM: You are a helpful assistant who has access to the following function to help the user, which you can use if needed -
```json
{
  "name": "search_wandb_training_run",
  "description": "Retrieve a wandb training run",
  "parameters": {
    "type": "object",
    "properties": {
      "model_name": {
        "type": "string",
        "description": "Name of the model to be retrieved"
      },
      "created_at": {
        "type": "string",
        "description": "data of creation of the training run"
      }
    },
    "required": [
      "model_name"
    ]
  }
}
```
USER: I'd like to retrieve information corresponding to my llama2 7b fine-tuning run which I created yesterday.
ASSISTANT: <functioncall>
```json
{
    "model_name": "llama2 7b",
    "created_at": "2023-11-16 19:02:47.169823"
}
```
FUNCTION RESPONSE:
```json
{
    "model_id": "llama2-ft-522780665450614404",
    "created_at": "2023-11-15 12:13:30.169383",
    "epoch": 3.4,
    "loss": 1.12,
    ...
}
```
ASSISTANT: The model you are looking for has id `llama2-ft-522780665450614404`. It finished training yesterday at 4pm after completing 3.4 epochs...
```

In this setup we use GPT to infer the inner narrative motivating the user's actions.

This format also was inspired by the [Glaive function calling dataset](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2).

## Quick Start

This is made to run on macos and has only been tested on Monterey 12.3.1. Setup your python environment. Make sure you have an openai api key to use the `compile.py` module.

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Place your `OPENAI_API_KEY` in a `.env` file at the root of the cloned repository.

## Record Activity

Record user clicks, mouse movements, and keystrokes with `record.py`. Screenshots are taken on every clicks, at intervals of 30 seconds minimum. Launch the recording:
```bash
python record.py
```

All recorded data is put in a `data/` folder. No compression is applied on the produced artifacts (i.e. this is inefficient and will take space).

## Extract Dataset

Format the data into text with `compile.py`. This parses the recorded screenshots & user entries into activity time intervals, and extract all produced and consumed text within each interval. The openai api is then used to reproduce the user workflow formatted as API calls.

Compile text samples from the recorded data by running:
```bash
python compile.py
```
> Optionally, add the `--use-vision` flag to use the `gpt-4-vision-preview` model to extract the API signature directly from the screenshot instead of using extract text.

## Other Features

At first I thought a lot of this could be done without deep learning. I wrote some code to run edge detection on each screenshot and narrow down the text *in context* for the user (by looking at their mouse's position and finding the most-central window corresponding to this position).

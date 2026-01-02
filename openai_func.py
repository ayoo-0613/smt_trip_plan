import requests
import os
import numpy as np
import copy
import ast
import re
import math
import time
import anthropic
import openai

def _trim_text(text, limit=500):
  text = text or ""
  if len(text) <= limit:
    return text
  return text[:limit] + "...(truncated)"

def _ollama_debug_log(section, content):
  log_path = os.environ.get("OLLAMA_DEBUG_LOG")
  if not log_path:
    return
  try:
    with open(log_path, "a") as f:
      f.write(f"[{section}]\n{content}\n\n")
  except Exception:
    pass

def _normalize_messages(messages):
  if isinstance(messages, list):
    return messages
  return [{"role": "user", "content": str(messages)}]

def _resolve_local_model_name(model_name):
  if not model_name or model_name == "local":
    return os.environ.get("OLLAMA_MODEL", "llama2")
  if model_name.startswith("ollama:"):
    stripped = model_name.split("ollama:", 1)[1].strip()
    return stripped or os.environ.get("OLLAMA_MODEL", "llama2")
  return model_name

def _ollama_chat_response(messages, model_name, timeout=999):
  base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
  url = f"{base_url}/v1/chat/completions"
  api_key = os.environ.get("OLLAMA_API_KEY") or os.environ.get("OPENAI_API_KEY")
  headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
  payload = {
    "model": model_name,
    "messages": messages,
    "temperature": 0.0,
    "stream": False,
  }
  _ollama_debug_log("ollama_chat_request", f"url={url}\nmodel={model_name}\n")
  resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
  if not resp.ok:
    raise RuntimeError(f"Ollama chat error {resp.status_code}: {_trim_text(resp.text)}")
  try:
    data = resp.json()
  except Exception:
    raise RuntimeError(f"Ollama chat non-JSON response: {_trim_text(resp.text)}")
  _ollama_debug_log("ollama_chat_response", _trim_text(resp.text))
  return data["choices"][0]["message"]["content"]

def _ollama_generate_response(prompt, model_name, timeout=999):
  base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
  url = f"{base_url}/api/generate"
  payload = {
    "model": model_name,
    "prompt": prompt,
    "stream": False
  }
  _ollama_debug_log("ollama_generate_request", f"url={url}\nmodel={model_name}\n")
  resp = requests.post(url, json=payload, timeout=timeout)
  if not resp.ok:
    raise RuntimeError(f"Ollama generate error {resp.status_code}: {_trim_text(resp.text)}")
  try:
    data = resp.json()
  except Exception:
    raise RuntimeError(f"Ollama generate non-JSON response: {_trim_text(resp.text)}")
  _ollama_debug_log("ollama_generate_response", _trim_text(resp.text))
  return data.get("response", "")

def Local_response(prompt, model_name="llama2"):
  model_name = _resolve_local_model_name(model_name)
  messages = _normalize_messages(prompt)
  if isinstance(prompt, list):
    prompt_text = "\n".join([item.get("content", "") for item in prompt])
  else:
    prompt_text = prompt
  chat_error = None
  try:
    return _ollama_chat_response(messages, model_name)
  except Exception as e:
    chat_error = e
    try:
      return _ollama_generate_response(prompt_text, model_name)
    except Exception as e:
      return f"[Local LLM Error] chat={chat_error}; generate={e}"


claude_api_key_name = ...# your key
mixtral_api_key_name = ...# your key

def GPT_response(messages, model_name):
  if model_name in ['gpt-4-turbo-preview','gpt-4-1106-preview', 'gpt-4', 'gpt-4o', 'gpt-4-32k', 'gpt-3.5-turbo-0301', 'gpt-4-0613', 'gpt-4-32k-0613', 'gpt-3.5-turbo-16k-0613', 'gpt-3.5-turbo']:
    #print(f'-------------------Model name: {model_name}-------------------')
    response = openai.ChatCompletion.create(
      model=model_name,
      messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": messages}
        ],
      temperature = 0.0,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    return response.choices[0].message.content
  return Local_response(messages, model_name=model_name)

def Claude_response(messages):
  client = anthropic.Anthropic(
    api_key=claude_api_key_name,
  )
  message = client.messages.create(
    model="claude-3-opus-20240229", # claude-3-sonnet-20240229, claude-3-opus-20240229, claude-3-haiku-20240307
    max_tokens=4096,
    temperature=0.0,
    system="",
    messages=[
        {"role": "user", "content": messages}
    ]
  )
  return message.content[0].text

def Mixtral_response(messages, mode = 'normal'):
  model = 'mistral-large-latest'
  client = MistralClient(api_key=mixtral_api_key_name)

  if mode == 'json':
    messages = [
        ChatMessage(role="system", content="You are a helpful code assistant. Your task is to generate a valid JSON object based on the given information. Please only produce the JSON output and avoid explaining."), 
        ChatMessage(role="user", content=messages)
    ]
  elif mode == 'code':
    messages = [
    ChatMessage(role="system", content="You are a helpful code assistant that help with writing Python code for a user requests. Please only produce the function and avoid explaining. Do not add \ in front of _"),
    ChatMessage(role="user", content=messages)
  ]
  else: 
    messages = [
    ChatMessage(role="user", content=messages)
  ]

  # No streaming
  chat_response = client.chat(
      model=model,
      messages=messages,
      temperature=0.0,
  )
  # import pdb; pdb.set_trace()
  return chat_response.choices[0].message.content

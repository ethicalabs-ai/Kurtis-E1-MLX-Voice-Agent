def get_llm_response(text, client, history, llm_model, max_tokens):
    history.append({"role": "user", "content": text})
    response = client.chat.completions.create(
        model=llm_model,
        messages=history,
        max_tokens=max_tokens,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
            }
        },
    )
    choice = response.choices[0]
    assistant_response = choice.message.content

    if not assistant_response or not assistant_response.strip():
        raise ValueError("LLM returned an empty response")

    assistant_response = assistant_response.strip()
    history.append({"role": "assistant", "content": assistant_response})
    return assistant_response


def translate_text(
    text,
    client,
    from_language,
    to_language,
    config,
    translation_model,
    max_tokens,
    temperature=0.0,
):
    from_lang_str = config.SUPPORTED_LANGUAGES[from_language]["name"]
    to_lang_str = config.SUPPORTED_LANGUAGES[to_language]["name"]
    prompt = f"""Translate the following {from_lang_str} source text to {to_lang_str}:
{from_lang_str}: {text}
{to_lang_str}: """
    translation = client.completions.create(
        model=translation_model,
        prompt=prompt,
        max_tokens=256,
        temperature=temperature,
        best_of=1,
        stop="<eos>",
    )
    text = translation.choices[0].text.strip()
    return text

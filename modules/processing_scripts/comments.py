from modules import shared
import re

def strip_comments(text):
    if shared.opts.enable_prompt_comments:
        text = re.sub('(^|\n)#[^\n]*(\n|$)', '\n', text)    # whole line comment
        text = re.sub('#[^\n]*(\n|$)', '\n', text)          # in the middle of the line comment

    return text

shared.options_templates.update(shared.options_section(('sd', "Stable Diffusion", "sd"), {
    "enable_prompt_comments": shared.OptionInfo(True, "Enable comments").info("Use # anywhere in the prompt to hide the text between # and the end of the line from the generation."),
}))

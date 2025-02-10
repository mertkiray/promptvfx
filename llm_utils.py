from __future__ import annotations

import os
from typing import Any

from openai import OpenAI, NotGiven
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class TimeSegment(BaseModel):
    segment_title: str
    ends_after: float
    centers_effect: str
    rgbs_effect: str
    opacities_effect: str

class AnimationSummary(BaseModel):
    time_segments: list[TimeSegment]
    animation_title: str

def prompt_4o(prompt: str, system_message: str, response_format: Any | None = None, temperature: float = 1.0):
    completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    temperature=temperature,
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ],
    response_format=response_format or NotGiven(),
    )
    
    if response_format:
        return completion.choices[0].message.parsed or ""
    
    
    return completion.choices[0].message.content or ""
    
        

def extract_centers_summary(animation_summary):
    centers_summary = ""
    for time_segment in animation_summary.time_segments:
        centers_summary += f"## {time_segment.segment_title} (t<={time_segment.ends_after}s)"
        centers_summary += "\n\n"
        centers_summary += f"- {time_segment.centers_effect}"
        centers_summary += "\n\n"
    return centers_summary

def extract_rgbs_summary(animation_summary):
    rgbs_summary = ""
    for time_segment in animation_summary.time_segments:
        rgbs_summary += f"## {time_segment.segment_title} (t<={time_segment.ends_after}s)"
        rgbs_summary += "\n\n"
        rgbs_summary += f"- {time_segment.rgbs_effect}"
        rgbs_summary += "\n\n"
    return rgbs_summary

def extract_opacities_summary(animation_summary):
    opacities_summary = ""
    for time_segment in animation_summary.time_segments:
        opacities_summary += f"## {time_segment.segment_title} (t<={time_segment.ends_after}s)"
        opacities_summary += "\n\n"
        opacities_summary += f"- {time_segment.opacities_effect}"
        opacities_summary += "\n\n"
    return opacities_summary

def prompt_o1(prompt: str) -> str:
    completion = client.chat.completions.create(
    model="o1-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ],
    )

    return completion.choices[0].message.content or ""
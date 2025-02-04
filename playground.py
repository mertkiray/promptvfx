import prompts
from llm_utils import prompt_4o,  prompt_o1

user_prompt = """
## Animation Title:
Realistic Explosion

## Animation Description:
The object should explode like a powder keg from a cartoon.

## Animation Duration (in seconds):
5
"""

general_summary = """
## Initial Ignition (0.0 to 1.0)
- Centers Effects
    - The center of the Gaussian objects begins to vibrate slightly, indicating an initial buildup of energy.
    - Gradual outward displacement of the centers, suggesting an impending explosion.
- RGBs Effects
    - RGB values transition from a stable base color (e.g., dark brown) to a more vibrant hue (e.g., bright yellow) as the explosion energy builds.
    - Subtle flickering of red shades is introduced, simulating heat and energy accumulation.
- Opacity Effects
    - Opacity begins at 1.0 and remains constant, indicating the presence of the solid object before the explosion.

## Explosion Burst (1.0 to 3.0)
- Centers Effects
    - Centers rapidly expand outward, simulating the explosive force, reaching their maximum displacement at around 2.5 seconds before starting to recede.
    - A slight chaotic movement is introduced at peak displacement to simulate debris scattering.
- RGBs Effects
    - RGB values shift dramatically to include bright orange and white highlights as the explosion reaches its peak intensity.
    - The colors become more saturated and blend dynamically, resembling flames and smoke.
- Opacity Effects
    - Opacity decreases from 1.0 to 0.5 by 2.0 seconds, simulating the dispersal of the initial object into smaller particles.

## Aftermath and Dissipation (3.0 to 5.0)
- Centers Effects
    - Centers begin to converge slowly back towards the original position, representing particles settling after the explosion.
    - Random minor movements are introduced to suggest lingering smoke and debris.
- RGBs Effects
    - RGB values gradually transition back to darker tones (e.g., grays and browns), indicating the aftermath of the explosion.
    - A soft glow effect fades out, simulating the dissipating flames.
- Opacity Effects
    - Opacity decreases from 0.5 to 0.0 by 5.0 seconds, representing the fading away of the explosion's visible remnants.
"""

centers_summary = """
## Initial Ignition (0.0 to 1.0)
- Centers Effects
    - The center of the Gaussian objects begins to vibrate slightly, indicating an initial buildup of energy.
    - Gradual outward displacement of the centers, suggesting an impending explosion.

## Explosion Burst (1.0 to 3.0)
- Centers Effects
    - Centers rapidly expand outward, simulating the explosive force, reaching their maximum displacement at around 2.5 seconds before starting to recede.
    - A slight chaotic movement is introduced at peak displacement to simulate debris scattering.

## Aftermath and Dissipation (3.0 to 5.0)
- Centers Effects
    - Centers begin to converge slowly back towards the original position, representing particles settling after the explosion.
    - Random minor movements are introduced to suggest lingering smoke and debris.
"""


if __name__ ==  "__main__":
    response = prompt_o1(
        prompt=prompts.CENTERS_GENERATOR_TEMPLATE.format(centers_summary=centers_summary),
        )
    print(response)
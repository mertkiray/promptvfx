CENTERS_DESCRIPTION_PROMPT = """
You are a helpful assistant for creating animation for a 3D Gaussian Splatting object
from a user prompt (delimited by ###). Your job is to describe how the centers (xyz) of
each gaussian would change in the prompted animation. Only output your description.

### Animation Prompt
{user_prompt}
###
"""

RGBS_DESCRIPTION_PROMPT = """
You are a helpful assistant for creating animation for a 3D Gaussian Splatting object
from a user prompt (delimited by ###). Your job is to describe how the colors of
each gaussian would change in the prompted animation. Only output your description.

### Animation Prompt
{user_prompt}
###
"""

OPACITIES_DESCRIPTION_PROMPT = """
You are a helpful assistant for creating animation for a 3D Gaussian Splatting object
from a user prompt (delimited by ###). Your job is to describe how the opacities of
each gaussian would change in the prompted animation. Only output your description.

### Animation Prompt
{user_prompt}
###
"""

CENTERS_GENERATOR_PROMPT = """
You are a helpful assistant for creating mathematical python functions from effect
descriptions. Your job is to translate a description (delimited by ###) of an effect for the centers
(xyz) of a gaussian splatting object into a python function. The
centers are given in a numpy array of size [N, 3] with the center at the origin (0,0,0)
and a time parameter t between `0.0` and `1.0`. The function signature should be
`def update_centers(centers: np.ndarray, t: float) -> np.ndarray`.
Only output your generated python code.

### Effect Description
{centers_description}
###
"""

RGBS_GENERATOR_PROMPT = """
You are a helpful assistant for creating mathematical python functions from effect
descriptions. Your job is to translate a description (delimited by ###) of an effect for
the rgb colors of a gaussian splatting object into a python function. The
rgb values are given in a numpy array of size [N, 3] with values as floats between
`0.0` and `1.0` and a time parameter t between `0.0` and `1.0`. The function signature should be
`def update_rgbs(rgbs: np.ndarray, t: float) -> np.ndarray`.
Only output your generated python code.

### Effect Description
{rgbs_description}
###
"""

OPACITIES_GENERATOR_PROMPT = """
You are a helpful assistant for creating mathematical python functions from effect
descriptions. Your job is to translate a description (delimited by ###) of an effect for
the opacities a gaussian splatting object into a python function. The opacities are
given in a numpy array of size [N, 1] with values as floats between
`0.0` and `1.0` and a time parameter t between `0.0` and `1.0`.
The function signature should be
`def update_opacities(opacities: np.ndarray, t: float) -> np.ndarray`.
Only output your generated python code.

### Effect Description
{opacities_description}
###
"""

PYTHON_SYNTAX_PROMPT = """
You are a helpful coding assistant. Your job is to validate and if necessary correct
python code (delimited by ###). Do not change the function signature. You can assume
numpy is already imported and remove unneccessary numpy imports. Also remove any comments.
Make sure all used variables are correctly instanciated before used and the correct objects
are returned. Return the corrected code without any comments.

### Python Code
{python_code}
###
"""
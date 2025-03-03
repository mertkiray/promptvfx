from animation import Animation


# Example Explosion

EXPLOSION_CENTERS_CODE = '''
def compute_centers(t: float, centers: np.ndarray) -> np.ndarray:
    """
    Compute the updated center positions of Gaussians in a Gaussian splatting object at a given time.

    This function calculates the new positions of `N` Gaussians in an animation, based on the 
    provided time `t`. The input `centers` array has a shape of [N, 3], where each row represents 
    the (x, y, z) coordinates of a Gaussian
    where +X is forward, -X is backward, +Y is left, -Y is right, +Z is up, and -Z is down.
    The function applies numpy array operations to update the positions dynamically over time.

    Args:
        t (float): The current time in the animation, ranging from `0.0` to `1.0`.
        centers (np.ndarray): A NumPy array of shape [N, 3], where each row contains the 
                              (x, y, z) coordinates of a Gaussian.

    Returns:
        np.ndarray: A NumPy array of shape [N, 3], containing the updated center positions 
                    of the Gaussians at time `t`.
    """
    N = centers.shape[0]
    new_centers = centers.copy()

    if t <= 0.3:
        scale_factor = 1.1  # 10% scale increase
        new_centers *= scale_factor
    elif t <= 0.7:
        # Recursive call to get the centers from the last second of Phase 1
        new_centers = compute_centers(0.3, centers)
        scale_factor = 4.0  # 300% increase
        direction = np.random.uniform(-1, 1, new_centers.shape)  # random direction for explosion
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)  # normalize
        new_centers += direction * scale_factor * (t - 0.3) / 0.4  # disperse based on time
    else:
        # Phase 3: Keep the centers from Phase 2
        new_centers = compute_centers(0.7, centers)

    return new_centers
'''.strip()

EXPLOSION_RGBS_CODE = '''
def compute_rgbs(t: float, rgbs: np.ndarray) -> np.ndarray:
    """
    Computes the colors of Gaussians in a Gaussian splatting object at a given point in time during an animation.

    This function updates the RGB colors of N Gaussians based on the time parameter `t`. The input `rgbs` array 
    contains the initial colors of the Gaussians, where each row represents a color (r, g, b) with values in 
    the range [0.0, 1.0]. The function applies numpy operations to compute a transformed color array that 
    reflects the color changes over time.

    Args:
        t (float): The current time in the animation, ranging from `0.0` to `1.0`.
        rgbs (np.ndarray): A NumPy array of shape (N, 3) representing the initial colors of N Gaussians.

    Returns:
        np.ndarray: A NumPy array of shape (N, 3) containing the updated colors at time `t`.
    """
    N = rgbs.shape[0]
    new_rgbs = rgbs.copy()

    # Define constants for color transformations
    BRIGHTNESS_FACTOR = 1.5  # Factor to brighten colors

    # Phase 1: Initial Build-Up (0.0 - 0.3 seconds)
    if t <= 0.3:
        new_rgbs = np.clip(new_rgbs * BRIGHTNESS_FACTOR, 0.0, 1.0)

    # Phase 2: Outwards Explosion (0.3 - 0.7 seconds)
    elif t <= 0.7:
        new_rgbs = compute_rgbs(0.3, rgbs)  # Get colors from Phase 1

    # Phase 3: Further Explosion and Gravity Fall Down (0.7 - 1.0 seconds)
    else:
        new_rgbs = compute_rgbs(0.7, rgbs)  # Get colors from Phase 2

    return new_rgbs
'''.strip()

EXPLOSION_OPACITIES_CODE = '''
def compute_opacities(t: float, opacities: np.ndarray) -> np.ndarray:
    """
    Compute the updated opacity values for the Gaussians in a Gaussian splatting object 
    at a given point in time during an animation.

    The function takes the current time `t` within the animation and an array of opacity 
    values representing the transparency of each Gaussian. It then applies numpy-based 
    operations to compute the updated opacities for the given time step.

    Args:
        t (float): The current time in the animation, ranging from `0.0` to `1.0`.
        opacities (np.ndarray): A 1D numpy array of shape (N, 1) containing opacity values 
                                 for N Gaussians, where each value is between `0.0` (fully 
                                 transparent) and `1.0` (fully opaque).

    Returns:
        np.ndarray: A 1D numpy array of shape (N, 1) with updated opacity values computed 
                    based on the provided time `t`.
    """
    N = opacities.shape[0]
    new_opacities = opacities.copy()

    # Define constants for opacity transformations
    phase_1_end = 0.3
    phase_2_end = 0.7
    
    if t <= phase_1_end:
        # Phase 1: Initial Build-Up
        new_opacities = new_opacities * (t / phase_1_end)
    elif t <= phase_2_end:
        # Phase 2: Outwards Explosion
        # Maintain the same opacities as the end of Phase 1
        new_opacities = opacities * (1.0)  # No change, keep max opacities from Phase 1
    else:
        # Phase 3: Further Explosion and Gravity Fall Down
        # Recursively get opacities at the end of Phase 2
        new_opacities = compute_opacities(phase_2_end, opacities)
        # Decrease opacities to create a fading effect
        new_opacities = new_opacities * (1 - (t - phase_2_end) / (1.0 - phase_2_end))

    return new_opacities.reshape(N, 1)
'''.strip()

EXAMPLE_EXPLOSION = Animation(
    title="Example Explosion",
    duration=1,
    centers_code=EXPLOSION_CENTERS_CODE,
    rgbs_code=EXPLOSION_RGBS_CODE,
    opacities_code=EXPLOSION_OPACITIES_CODE,
)

# Example Color Shift

COLOR_SHIFT_CENTERS_CODE = '''
def compute_centers(t: float, centers: np.ndarray) -> np.ndarray:
    """
    Compute the updated center positions of Gaussians in a Gaussian splatting object at a given time.

    This function calculates the new positions of `N` Gaussians in an animation, based on the 
    provided time `t`. The input `centers` array has a shape of [N, 3], where each row represents 
    the (x, y, z) coordinates of a Gaussian
    where +X is forward, -X is backward, +Y is left, -Y is right, +Z is up, and -Z is down.
    The function applies numpy array operations to update the positions dynamically over time.

    Args:
        t (float): The current time in the animation, ranging from `0.0` to `4.0`.
        centers (np.ndarray): A NumPy array of shape [N, 3], where each row contains the 
                              (x, y, z) coordinates of a Gaussian.

    Returns:
        np.ndarray: A NumPy array of shape [N, 3], containing the updated center positions 
                    of the Gaussians at time `t`.
    """
    N = centers.shape[0]
    new_centers = centers.copy()

    if t <= 0.5:
        return new_centers  # Phase 1

    elif t <= 1.5:
        return new_centers  # Phase 2

    elif t <= 2.5:
        return new_centers  # Phase 3

    elif t <= 3.5:
        return new_centers  # Phase 4

    elif t <= 4.0:
        return new_centers  # Phase 5

    return new_centers
'''.strip()

COLOR_SHIFT_RGBS_CODE = '''
def compute_rgbs(t: float, rgbs: np.ndarray) -> np.ndarray:
    """
    Computes the colors of Gaussians in a Gaussian splatting object at a given point in time during an animation.

    This function updates the RGB colors of N Gaussians based on the time parameter `t`. The input `rgbs` array 
    contains the initial colors of the Gaussians, where each row represents a color (r, g, b) with values in 
    the range [0.0, 1.0]. The function applies numpy operations to compute a transformed color array that 
    reflects the color changes over time.

    Args:
        t (float): The current time in the animation, ranging from `0.0` to `4.0`.
        rgbs (np.ndarray): A NumPy array of shape (N, 3) representing the initial colors of N Gaussians.

    Returns:
        np.ndarray: A NumPy array of shape (N, 3) containing the updated colors at time `t`.
    """
    N = rgbs.shape[0]
    new_rgbs = rgbs.copy()

    # Define color transformation constants
    def transition(t, start_color, end_color, start_time, end_time):
        if start_time <= t <= end_time:
            fraction = (t - start_time) / (end_time - start_time)
            return start_color + fraction * (end_color - start_color)
        return None

    # Phase 1: Keep colors unchanged
    if t <= 0.5:
        return new_rgbs

    # Phase 2: Color Transition Begins
    if t <= 1.5:
        if t <= 0.75:
            color_transform = transition(t, np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.5, 0.75)
        elif t <= 1.0:
            color_transform = transition(t, np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.5, 0.0]), 0.75, 1.0)
        elif t <= 1.25:
            color_transform = transition(t, np.array([1.0, 0.5, 0.0]), np.array([1.0, 1.0, 0.0]), 1.0, 1.25)
        else:  # t <= 1.5
            color_transform = transition(t, np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]), 1.25, 1.5)

        if color_transform is not None:
            new_rgbs = new_rgbs * (1 - (t % 0.5) / 0.5) + color_transform * (t % 0.5 / 0.5)

    # Phase 3: Continued Color Transition
    elif t <= 2.5:
        if t <= 1.75:
            color_transform = transition(t, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), 1.5, 1.75)
        elif t <= 2.0:
            color_transform = transition(t, np.array([0.0, 0.0, 1.0]), np.array([0.29, 0.0, 0.51]), 1.75, 2.0)
        elif t <= 2.25:
            color_transform = transition(t, np.array([0.29, 0.0, 0.51]), np.array([0.56, 0.0, 1.0]), 2.0, 2.25)
        else:  # t <= 2.5
            color_transform = transition(t, np.array([0.56, 0.0, 1.0]), np.array([0.4, 0.0, 0.56]), 2.25, 2.5)

        if color_transform is not None:
            new_rgbs = new_rgbs * (1 - (t % 0.5) / 0.5) + color_transform * (t % 0.5 / 0.5)

    # Phase 4: Reverse Color Transition
    elif t <= 3.5:
        if t <= 2.75:
            color_transform = transition(t, np.array([0.4, 0.0, 0.56]), np.array([0.29, 0.0, 0.51]), 2.5, 2.75)
        elif t <= 3.0:
            color_transform = transition(t, np.array([0.29, 0.0, 0.51]), np.array([0.0, 0.0, 1.0]), 2.75, 3.0)
        elif t <= 3.25:
            color_transform = transition(t, np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), 3.0, 3.25)
        else:  # t <= 3.5
            color_transform = transition(t, np.array([0.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0]), 3.25, 3.5)

        if color_transform is not None:
            new_rgbs = new_rgbs * (1 - (t % 0.5) / 0.5) + color_transform * (t % 0.5 / 0.5)

    # Phase 5: Return to Original State
    elif t <= 4.0:
        if t <= 3.75:
            color_transform = transition(t, np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.5, 0.0]), 3.5, 3.75)
        else:  # t <= 4.0
            color_transform = transition(t, np.array([1.0, 0.5, 0.0]), np.array([1.0, 0.0, 0.0]), 3.75, 4.0)

        if color_transform is not None:
            new_rgbs = new_rgbs * (1 - (t % 0.25) / 0.25) + color_transform * (t % 0.25 / 0.25)

    return new_rgbs
'''.strip()

COLOR_SHIFT_OPACITIES_CODE = '''
def compute_opacities(t: float, opacities: np.ndarray) -> np.ndarray:
    """
    Compute the updated opacity values for the Gaussians in a Gaussian splatting object 
    at a given point in time during an animation.

    The function takes the current time `t` within the animation and an array of opacity 
    values representing the transparency of each Gaussian. It then applies numpy-based 
    operations to compute the updated opacities for the given time step.

    Args:
        t (float): The current time in the animation, ranging from `0.0` to `4.0`.
        opacities (np.ndarray): A 1D numpy array of shape (N, 1) containing opacity values 
                                 for N Gaussians, where each value is between `0.0` (fully 
                                 transparent) and `1.0` (fully opaque).

    Returns:
        np.ndarray: A 1D numpy array of shape (N, 1) with updated opacity values computed 
                    based on the provided time `t`.
    """
    N = opacities.shape[0]
    new_opacities = opacities.copy()

    if t <= 0.5:
        return new_opacities  # Phase 1
    elif t <= 1.5:
        return new_opacities  # Phase 2
    elif t <= 2.5:
        return new_opacities  # Phase 3
    elif t <= 3.5:
        return new_opacities  # Phase 4
    else:
        return new_opacities  # Phase 5
'''.strip()

EXAMPLE_COLOR_SHIFT = Animation(
    title="Example Color Shift",
    duration=4,
    centers_code=COLOR_SHIFT_CENTERS_CODE,
    rgbs_code=COLOR_SHIFT_RGBS_CODE,
    opacities_code=COLOR_SHIFT_OPACITIES_CODE,
)
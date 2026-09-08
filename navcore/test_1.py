"""Quick manual visual check: decompose the built environment's free
space and render the resulting convex cells on top of the obstacles,
crowd, and robot.

Run directly:

    python -m navcore.test_show_cells
"""

import matplotlib.pyplot as plt

from navcore.boustropheden.boustropheden import decompose
from navcore.builder.environment_builder import EnvironmentBuilder
from navcore.visualization.visualizer import Visualizer


def main() -> None:
    env = EnvironmentBuilder().build_environment()
    print(env.__annotations__)
    result = decompose(env)
    print(result)
    visualizer = Visualizer()
    visualizer.visualize(env, decomposition=result)
    plt.show()


if __name__ == "__main__":
    main()

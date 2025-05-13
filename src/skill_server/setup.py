from setuptools import setup, find_packages
from glob import glob
import os

package_name = "skill_server"
module_name = "skill_server/resource"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name, module_name],
    # packages=[package_name
    #           ],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # (
        #     "share/" + package_name + "/resource",
        # ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="blackbird",
    maintainer_email="zwh@163.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vla_service = skill_server.vla_service:main",
         ],
    },
)

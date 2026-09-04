from setuptools import find_packages, setup

package_name = "smart_wheelchair_safety"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SmartWheelChair",
    maintainer_email="dev@example.com",
    description="Shared-control safety filter for the smart wheelchair simulator.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "safety_filter_node = smart_wheelchair_safety.safety_filter_node:main",
            "web_joystick_node = smart_wheelchair_safety.web_joystick_node:main",
        ],
    },
)

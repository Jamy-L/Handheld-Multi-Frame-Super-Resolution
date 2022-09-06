import setuptools

setuptools.setup(
    name='fast_optics_correction',
    version="1.0.0",
    author="Thomas Eboli",
    author_email="thomas.eboli@ens-paris-saclay.fr",
    description="Fast two-step blind optical aberration correction [ECCV22]",
    url="https://github.com/teboli/fast_two_stage_psf_correction",
    packages = setuptools.find_packages(exclude=["training"]),
    include_package_data=True,
)

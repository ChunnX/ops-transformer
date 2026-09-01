# Auto-generated script for creating setup.py
file(WRITE "/home/developer/codes/ops-transformer/build/prepare_build/es_transformer_cust_build/python_package/setup.py" "from setuptools import setup, find_packages

setup(
    name='es_transformer_cust',
    version='1.0.0',
    description='ES Generated API for es_transformer_cust operators',
    author='Huawei Technologies Co., Ltd.',
    packages=find_packages(),
    python_requires='>=3.7',
    entry_points={
        'ge.es.plugins': [
            'transformer_cust = es_transformer_cust:get_module',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
")
message(STATUS "Created setup.py for package 'es_transformer_cust'")

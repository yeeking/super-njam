# Coding guidelines

When writing python, it is fine to create a virtual environment and work in that.

When writing python, do strong asserts which exit the script with helpful messages when the wrong or invalid arguments and parameters are passed in. 

Try to keep the python code modular: prefer organising code into loosely coupled functions that are easily testable and understandable to having long scripts with deeply nested levels. 

Assert data formats and shapes are correct for training and data processing - fail quickly so can debug and fix

Run and test code - if it is python, run short smoke tests on scripts and verify data is being processed and generated correctly. If c++, use cmake to build projects and check outputs for errors, fixing them as you go. 

Keep C++ code module with cohesive modules with clear purpose. Use classes defined in header files with implementation in cpp files. Break complex functions into calls to loosely coupled sub functions instead of making monolithic mega functions.

Maintain a README.md with minimal information on it with example bash commands to run all the parts of the project. Update this as you go.

Maintain a log of the project in log.md which explains which stage you are working on and some brief catch up notes. This will help me check how you are getting on and allow you to catch up with what you've done previously in case the context is wiped. Do not pollute the log with error fixing and very detailed information - just the key facts and stages and where we are at and what we are planning to do next. 

Please use planning mode and ask me for confirmation of complex decisions you need to make. 




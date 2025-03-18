# Before running, please add the project to your PYTHONPATH. For example:
# export PYTHONPATH=$PYTHONPATH:/research/d1/gds/ytyang/yichengfeng/Echo-workload-tracer
# Use the provided shell script to execute this Python file.
# Each framework requires specific arguments. At this stage, focus on completing the PyTorch part 
# (refer to the default arguments in the test directory; in PyTorch mode, you need to specify the --mode argument as 
# either runtime_profiling or graph_profiling, corresponding to ..database and ..graph respectively).
# The default output path is /output/pytorch/workload_graph or /output/pytorch/workload_runtime; 
# the former contains graph information, while the latter contains operator runtime details.
# Replace all print statements with the logging module, and ensure the console output format aligns with other projects. 
# Add real-time runtime testing output for each operator (refer to Proteus for implementation). 
# Profiling code should be placed in timer.py.
# Rename timer.py to profiling_timer.py.
# Move transformer to utils folder.
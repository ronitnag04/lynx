"""Config-file generator for ProtoAcc sweeps.

This module produces Chisel/Scala Config classes for the ProtoAcc
verilator sweep and the Yosys synth sweep. It shares its parameter
tables with :mod:`hw_cost_model.defaults` so both pipelines pull from
one canonical source.
"""

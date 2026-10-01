"""SmartRoute synthetic connected-vehicle data simulator.

Pipeline (each stage is its own module):

    road_network -> routes -> vehicles -> traffic -> trips -> telemetry
                                                      |            |
                                              historical.py     live.py

The geography is synthetic; nothing here represents a real city.
"""

__version__ = "0.1.0"

"""Constants and Conversion Factors"""


STANDARD_PRESSURE = 101325  # Pa
"""Standard atmospheric pressure in Pascals (Pa)"""
STANDARD_PRESSURE_IMPERIAL = 14.696  # psi
"""Standard atmospheric pressure in pounds per square inch (psi)"""
STANDARD_TEMPERATURE = 288.7056  # K (15.6oC)
"""Standard temperature in Kelvin (K)"""
STANDARD_TEMPERATURE_IMPERIAL = 60.0  # °F
"""Standard temperature in Fahrenheit (°F)"""
OIL_THERMAL_EXPANSION_COEFFICIENT = 9.7e-4  # 1/K
"""Thermal expansion coefficient for oil (1/K)"""
WATER_THERMAL_EXPANSION_COEFFICIENT = 3.0e-4  # 1/K
"""Thermal expansion coefficient for water (1/K)"""
WATER_ISOTHERMAL_COMPRESSIBILITY = 4.6e-10  # 1/Pa
"""Isothermal compressibility of water at 15.6°C in 1/Pa"""
OIL_THERMAL_EXPANSION_COEFFICIENT_IMPERIAL = 5.39e-4  # 1/°F
"""Thermal expansion coefficient for oil in imperial units (1/°F)"""
WATER_THERMAL_EXPANSION_COEFFICIENT_IMPERIAL = 1.67e-4  # 1/°F
"""Thermal expansion coefficient for water in imperial units (1/°F)"""
WATER_ISOTHERMAL_COMPRESSIBILITY_IMPERIAL = 3.17e-6  # 1/psi
"""Isothermal compressibility of water at 15.6°C in 1/psi"""
STANDARD_WATER_DENSITY = 998.2  # kg/m³
"""Standard water density at 15.6°C in kg/m³"""
STANDARD_WATER_DENSITY_IMPERIAL = 62.37  # lb/ft³
"""Standard water density at 15.6°C in lb/ft³"""
STANDARD_AIR_DENSITY = 1.225  # kg/m³
"""Standard air density at 15.6°C in kg/m³"""
STANDARD_AIR_DENSITY_IMPERIAL = 0.0765  # lb/ft³
"""Standard air density at 15.6°C in lb/ft³"""
MOLECULAR_WEIGHT_WATER = 18.01528  # g/mol
"""Molecular weight of water in grams per mole (g/mol)"""
MOLECULAR_WEIGHT_CO2 = 44.01  # g/mol
"""Molecular weight of carbon dioxide in grams per mole (g/mol)"""
MOLECULAR_WEIGHT_N2 = 28.0134  # g/mol
"""Molecular weight of nitrogen in grams per mole (g/mol)"""
MOLECULAR_WEIGHT_METHANE = 16.04246  # g/mol
"""Molecular weight of methane in grams per mole (g/mol)"""
MOLECULAR_WEIGHT_NACL = 58.44  # g/mol
"""Molecular weight of sodium chloride (NaCl) in grams per mole (g/mol)"""
MOLECULAR_WEIGHT_AIR = 28.9644  # g/mol
"""Molecular weight of air in grams per mole (g/mol) or (lb/lb-mol)"""
PSI_TO_PA = 6894.76  # 1 psi = 6894.76 Pa
"""Conversion factor from psi to Pascals (Pa)"""
PA_TO_PSI = 1 / PSI_TO_PA  
"""Conversion factor from Pascals to psi"""
RANKINE_TO_KELVIN = 5 / 9  # T(K) = T(R) * 5/9
"""Conversion factor from Rankine to Kelvin"""
KELVIN_TO_RANKINE = 9 / 5  # T(R) = T(K) * 9/5
"""Conversion factor from Kelvin to Rankine"""
CP_TO_PA_S = 0.001  # 1 cP = 0.001 Pa.s (Pascal-second)
"""Conversion factor from centipoise to Pascal-seconds (Pa·s)"""
MD_TO_M2 = 9.869233e-16  # 1 mD = 9.869233e-16 m^2
"""Conversion factor from millidarcies to square meters (m²)"""
SCF_PER_STB_TO_M3_PER_M3 = 0.1781076  # 1 scf/STB = ~0.178 m3/m3 (gas/oil ratio)
"""Conversion factor from standard cubic feet per stock tank barrel to cubic meters per cubic meter"""
M3_PER_M3_TO_SCF_PER_STB = 1 / SCF_PER_STB_TO_M3_PER_M3  
"""Conversion factor from cubic meters per cubic meter to standard cubic feet per stock tank barrel"""
M3_PER_M3_TO_BBL_PER_SCF = 0.1781076
"""Conversion factor from cubic meters per cubic meter to barrels per standard cubic foot (BBL/scf)"""
BBL_PER_SCF_TO_M3_PER_M3 = 1 / M3_PER_M3_TO_BBL_PER_SCF  
"""Conversion factor from barrels per standard cubic foot to cubic meters per cubic meter (m³/m³)"""
M3_PER_M3_TO_BBL_PER_STB = 1.0
"""Conversion factor from cubic meters per cubic meter to barrels per stock tank barrel (BBL/STB)"""
BBL_PER_STB_TO_M3_PER_M3 = 1 / M3_PER_M3_TO_BBL_PER_STB  
"""Conversion factor from barrels per stock tank barrel to cubic meters per cubic meter (m³/m³)"""
M3_TO_SCF = 35.3147  # 1 m³ = 35.3147 scf
"""Conversion factor from cubic meters to standard cubic feet (scf)"""
SCF_TO_BBL = 0.005615  # 1 scf = 0.005615 BBL
"""Conversion factor from standard cubic feet to barrels (BBL)"""
BBL_TO_FT3 = 5.614583  # 1 barrel = 5.614583 ft³
"""Conversion factor from barrels to cubic feet (ft³)"""
FT3_TO_BBL = 1 / BBL_TO_FT3
"""Conversion factor from cubic feet to barrels (ft³)"""
STB_TO_FT3 = 5.614583  # 1 stock tank barrel = 5.614583 ft³
"""Conversion factor from stock tank barrels to cubic feet (ft³)"""
FT3_TO_STB = 1 / STB_TO_FT3  
"""Conversion factor from cubic feet to stock tank barrels (ft³)"""
STB_TO_M3 = 0.158987  # 1 stock tank barrel = 0.158987 m^3
"""Conversion factor from stock tank barrels to cubic meters (m³)"""
BBL_TO_M3 = 0.158987  # 1 barrel = 0.158987 m^3
"""Conversion factor from barrels to cubic meters (m³)"""
M3_TO_BBL = 1 / BBL_TO_M3  
"""Conversion factor from cubic meters to barrels (m³)"""
SCF_TO_SCM = 0.0283168  # 1 standard cubic foot = 0.0283168 standard cubic meter
"""Conversion factor from standard cubic feet to standard cubic meters (m³)"""
CENTIPOISE_TO_PA_S = 0.001  # 1 cP = 0.001 Pa.s (Pascal-second)
"""Conversion factor from centipoise to Pascal-seconds (Pa·s)"""
PA_S_TO_CENTIPOISE = 1000  # 1 Pa.s = 1000 cP
"""Conversion factor from Pascal-seconds to centipoise (cP)"""
IDEAL_GAS_CONSTANT = 8.31446261815324  # J/(mol·K)
"""Universal gas constant in J/(mol·K)"""
IDEAL_GAS_CONSTANT_SI = 8.31446261815324e-3  # kJ/(mol·K)
"""Universal gas constant in kJ/(mol·K)"""
IDEAL_GAS_CONSTANT_IMPERIAL = 10.73159  # ft³·psi/(lb·mol·°R)
"""Universal gas constant in ft³·psi/(lb·mol·°R)"""
POUNDS_PER_FT3_TO_KG_PER_M3 = 16.0185  # 1 lb/ft³ = 16.0185 kg/m³
"""Conversion factor from pounds per cubic foot to kilograms per cubic meter (kg/m³)"""
KG_PER_M3_TO_POUNDS_PER_FT3 = (
    1 / POUNDS_PER_FT3_TO_KG_PER_M3
)  
"""Conversion factor from kilograms per cubic meter to pounds per cubic foot (lb/ft³)"""
POUNDS_PER_FT3_TO_GRAMS_PER_CM3 = 0.00160185  # 1 lb/ft³ = 0.0160185 g/cm³
POUNDS_PER_MOLE_TO_GRAMS_PER_MOLE = 453.59237  # 1 lb/mol = 453.59237 g/mol
"""Conversion factor from pounds per pound-mole to grams per mole (g/mol)"""
GRAMS_PER_MOLE_TO_POUNDS_PER_MOLE = (
    1 / POUNDS_PER_MOLE_TO_GRAMS_PER_MOLE
)  
"""Conversion factor from grams per mole to pounds per pound-mole (lb/mol)"""
PPM_TO_GRAMS_PER_LITER = 1e-3  # 1 ppm = 1 gram per liter
"""Conversion factor from parts per million to grams per liter (g/L)"""
GRAMS_PER_LITER_TO_PPM = 1 / PPM_TO_GRAMS_PER_LITER  
"""Conversion factor from grams per liter to parts per million (ppm)"""
SCF_PER_POUND_MOLE = 379.49  # 1 lb-mol = 379.49 scf
"""Conversion factor from standard cubic feet to pound-moles (lb-mol)"""
INCHES_TO_METERS = 0.0254  # 1 inch = 0.0254 meters
"""Conversion factor from inches to meters (m)"""
METERS_TO_INCHES = 1 / INCHES_TO_METERS  
"""Conversion factor from meters to inches (m)"""
FT_TO_METERS = 0.3048  # 1 foot = 0.3048 meters
"""Conversion factor from feet to meters (m)"""
METERS_TO_FT = 1 / FT_TO_METERS  
"""Conversion factor from meters to feet (m)"""
M3_PER_SECOND_TO_STB_PER_DAY = 543168.384
"""Conversion factor from cubic meters per second to stock tank barrels per day (STB/day)"""
STB_PER_DAY_TO_M3_PER_SECOND = (
    1 / M3_PER_SECOND_TO_STB_PER_DAY
)  
"""Conversion factor from stock tank barrels per day to cubic meters per second (m³/s)"""
M3_PER_SECOND_TO_SCF_PER_DAY = 3049492.8
"""Conversion factor from cubic meters per second to standard cubic feet per day (scf/day)"""
SCF_PER_DAY_TO_M3_PER_SECOND = (
    1 / M3_PER_SECOND_TO_SCF_PER_DAY
)  
"""Conversion factor from standard cubic feet per day to cubic meters per second (m³/s)"""
DAYS_TO_SECONDS = 3600 * 24
"""Conversion factor from days to seconds"""
SECONDS_TO_DAYS = 1 / DAYS_TO_SECONDS
"""Conversion factor from seconds to days"""
MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY = 0.001127
"""Conversion factor from millidarcies per centipoise to ft²/psi.day"""
MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_SECOND = (
    MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY / DAYS_TO_SECONDS
)
PPM_TO_WEIGHT_FRACTION = 1e-6
"""Conversion factor from parts per million (ppm) to weight fraction"""
PPM_TO_WEIGHT_PERCENT = 1e-4  # 1 ppm = 0.0001 weight percent
"""Conversion factor from parts per million (ppm) to weight percent"""
WEIGHT_PERCENT_TO_PPM = 1 / PPM_TO_WEIGHT_PERCENT
"""Conversion factor from weight percent to parts per million (ppm)"""


RESERVOIR_OIL_NAME = (
    "n-Dodecane"  # Heavy oil, can be changed to any fluid supported by CoolProp
)
"""Default displaced fluid for simulations, can be changed to any fluid supported by CoolProp"""
RESERVOIR_GAS_NAME = "Methane"  # Default gas that exist with oil in the reservoir, can be changed to any fluid supported by CoolProp

# Outside this ranges, the fluid model may be non-reservoir like.
MIN_VALID_PRESSURE = 1.45  
MAX_VALID_PRESSURE = 14700.0
MIN_VALID_TEMPERATURE = 32.0
MAX_VALID_TEMPERATURE = 482.0

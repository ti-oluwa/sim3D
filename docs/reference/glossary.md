# Glossary

## Petroleum Engineering Terms

**API gravity**
: A measure of how heavy or light a petroleum liquid is compared to water. Defined as $\text{API} = \frac{141.5}{\gamma_o} - 131.5$ where $\gamma_o$ is the oil specific gravity. Higher API values indicate lighter oils. Light crude is typically above 31 API, medium crude is 22 to 31, and heavy crude is below 22.

**Aquifer**
: A body of water-bearing rock connected to the reservoir that provides pressure support as fluids are produced. Aquifers can be modeled analytically (using Carter-Tracy or Van Everdingen-Hurst models) or numerically (by extending the simulation grid into the water zone).

**Aquifer influx**
: The volume of water flowing from an aquifer into the reservoir over time in response to pressure decline. Measured in reservoir barrels (bbl).

**Black-oil model**
: A simplified thermodynamic model that describes reservoir fluids as three phases (oil, water, gas) with pressure-dependent properties. Gas can dissolve in oil (solution gas) but oil and water are immiscible with each other. This is the most widely used model for conventional reservoir simulation.

**Bottom-hole pressure (BHP)**
: The pressure measured at the bottom of a well at the depth of the producing formation. BHP is the pressure used in flow rate calculations and well control specifications. Measured in psi.

**Bottom water drive**
: A recovery mechanism where an aquifer located below the oil zone pushes water upward into the oil column as pressure declines, displacing oil toward the wells.

**Bubble point pressure**
: The pressure at which the first gas bubble forms in an oil that is initially at a higher pressure (undersaturated). Below the bubble point, gas comes out of solution and forms a free gas phase. Also called saturation pressure.

**Capillary pressure**
: The pressure difference across the interface between two immiscible fluids in a porous medium, caused by surface tension and wettability. Capillary pressure depends on the saturations of the phases and controls the distribution of fluids in the transition zones between fluid contacts.

**Compressibility**
: A measure of how much a fluid or rock changes volume in response to a change in pressure. Defined as $c = -\frac{1}{V}\frac{\partial V}{\partial P}$. Measured in psi$^{-1}$.

**Connate water**
: The water present in the pore spaces of a reservoir rock at the time of hydrocarbon accumulation. Connate water is held in place by capillary forces and is typically immobile during production. Also called irreducible water saturation ($S_{wc}$).

**Critical gas saturation**
: The minimum gas saturation at which gas becomes mobile in a porous medium. Below this saturation, gas is trapped in isolated bubbles and cannot flow. Typically 2% to 5% of pore volume.

**Dip (structural)**
: The angle of inclination of a rock layer relative to the horizontal. Structural dip affects gravity-driven flow and the position of fluid contacts. Measured in degrees.

**Edge water drive**
: A recovery mechanism where an aquifer located at the lateral edges of the reservoir supplies water that displaces oil toward the producing wells.

**Effective permeability**
: The permeability of a rock to a specific fluid phase when multiple phases are present. It is the product of absolute permeability and relative permeability: $k_{\text{eff}} = k \cdot k_r$.

**Formation volume factor (FVF)**
: The ratio of the volume of a fluid at reservoir conditions (pressure, temperature) to its volume at standard (surface) conditions. Oil FVF ($B_o$) is measured in bbl/STB. Gas FVF ($B_g$) is measured in ft³/SCF. Water FVF ($B_w$) is measured in bbl/STB.

**Gas-oil contact (GOC)**
: The depth at which the gas cap meets the oil zone. Above the GOC, the pore space is primarily filled with gas. Below it, the pore space is primarily filled with oil (and connate water).

**Gas-oil ratio (GOR)**
: The volume of gas produced per unit volume of oil produced, both measured at standard conditions. Measured in SCF/STB. Solution GOR ($R_s$) refers specifically to the gas dissolved in oil at reservoir conditions.

**Immiscible displacement**
: A displacement process in which the injected fluid and the displaced fluid remain as separate phases (they do not mix at the molecular level). Waterflooding is the most common example of immiscible displacement.

**IMPES**
: Implicit Pressure, Explicit Saturation. A numerical solution method that solves the pressure equation implicitly (unconditionally stable) and updates saturations explicitly (subject to CFL stability condition). This is the most common method for black-oil simulation because it balances stability and computational cost.

**Irreducible water saturation**
: See Connate water.

**Miscible displacement**
: A displacement process in which the injected fluid and the displaced fluid mix completely at the molecular level, eliminating the interface between them. Miscible displacement eliminates capillary trapping and can achieve very high recovery factors. Requires the injected fluid to be miscible with the oil, which depends on pressure, temperature, and composition.

**Minimum miscibility pressure (MMP)**
: The lowest pressure at which a particular injection gas achieves miscible displacement with a particular oil. Below the MMP, the displacement is immiscible. Above the MMP, it is miscible. Measured in psi.

**Mobility**
: The ratio of a phase's effective permeability to its viscosity: $\lambda = k_r / \mu$. Mobility determines how easily a phase flows through the porous medium. The mobility ratio between displacing and displaced fluids controls sweep efficiency.

**Oil-water contact (OWC)**
: The depth at which the oil zone meets the water zone. Below the OWC, the pore space is primarily filled with water.

**Permeability (absolute)**
: A measure of a rock's ability to transmit fluid, independent of the fluid properties. Measured in millidarcies (mD). Higher permeability means fluid flows more easily through the rock.

**Porosity**
: The fraction of the total rock volume that is pore space (voids). Effective porosity includes only the interconnected pores that can transmit fluid. Dimensionless, typically expressed as a decimal (0.20 = 20%).

**Relative permeability**
: The ratio of the effective permeability of a phase at a given saturation to the absolute permeability of the rock. Dimensionless, ranges from 0 to 1. Relative permeability curves describe how easily each phase flows as a function of saturation.

**Residual oil saturation**
: The fraction of oil that remains trapped in the pore spaces after displacement by water ($S_{orw}$) or gas ($S_{org}$). This oil is immobile and cannot be recovered by the displacing fluid. Typically 15% to 35% of pore volume for water flooding.

**Skin factor**
: A dimensionless number that accounts for the additional pressure drop (positive skin) or reduced pressure drop (negative skin) near the wellbore due to formation damage, perforation, or stimulation. Positive skin indicates damage; negative skin indicates stimulation (e.g., hydraulic fracture).

**Solution gas**
: Natural gas that is dissolved in oil at reservoir pressure and temperature. As pressure drops below the bubble point, solution gas comes out of the oil and forms a free gas phase. The amount of gas dissolved is described by the gas-oil ratio ($R_s$).

**Specific gravity**
: The ratio of a substance's density to the density of a reference substance. For oil, the reference is water at standard conditions ($\gamma_o = \rho_o / \rho_w$). For gas, the reference is air ($\gamma_g = M_g / M_{\text{air}}$).

**Transmissibility**
: A quantity that describes the flow capacity between two adjacent grid cells. It incorporates the permeability, cross-sectional area, and distance between cell centers: $T = \frac{k \cdot A}{\Delta x}$. Transmissibility modification is how faults and fractures affect flow.

**Undersaturated oil**
: Oil at a pressure above its bubble point, meaning all the gas that can dissolve in the oil at that temperature is dissolved. No free gas phase exists. As pressure drops below the bubble point, the oil becomes saturated and free gas appears.

**Water cut**
: The fraction of the total produced liquid that is water, measured at surface conditions. Water cut increases during a waterflood as the injected water breaks through to the producing wells. Expressed as a fraction (0 to 1) or percentage.

**Water drive**
: A natural recovery mechanism in which pressure support comes from water encroachment (from an aquifer or injected water), displacing oil toward the producing wells.

**Wettability**
: The preference of the rock surface for one fluid over another. In a water-wet rock, water preferentially contacts the rock surface and occupies the smaller pores. In an oil-wet rock, oil preferentially contacts the surface. Wettability strongly affects relative permeability and capillary pressure curves.

---

## Numerical and Simulation Terms

**Boundary condition**
: A mathematical specification of what happens at the edges of the simulation domain. Common types include no-flow (Neumann with zero flux), constant pressure (Dirichlet), and specified flux (Neumann with non-zero flux). Boundary conditions determine whether fluid can enter or leave the domain at its edges.

**CFL condition**
: The Courant-Friedrichs-Lewy condition, a stability requirement for explicit numerical schemes. The CFL number must be less than 1 for stability: $\text{CFL} = v \cdot \Delta t / \Delta x < 1$. This limits the maximum timestep for a given grid resolution and flow velocity.

**Convergence tolerance**
: The threshold that determines when an iterative solver has found a solution that is "good enough." Tighter tolerances produce more accurate solutions but require more iterations. In BORES, pressure tolerance defaults to $10^{-6}$ and saturation tolerance to $10^{-4}$.

**Deserialization**
: The process of reconstructing an object from its stored representation (dictionary, JSON, HDF5, etc.). The reverse of serialization. In BORES, deserialization uses the `load()` function or `from_dict()` method.

**Explicit scheme**
: A numerical method that computes the solution at the next timestep directly from the current timestep values, without solving a system of equations. Fast per step but subject to CFL stability limits.

**Grid cell**
: The fundamental spatial unit of the simulation domain. Each cell has a position, size, and a set of property values (pressure, saturations, porosity, permeability). Cells exchange fluid with their neighbors according to Darcy's law.

**Implicit scheme**
: A numerical method that computes the solution at the next timestep by solving a system of equations that couples the new and current timestep values. Unconditionally stable (no CFL limit) but more expensive per step and requires iterative solvers.

**Krylov method**
: A class of iterative methods for solving large sparse linear systems. Named after Alexei Krylov. Examples include Conjugate Gradient (CG), BiCGSTAB, and GMRES. These methods build a solution in a subspace that grows with each iteration.

**Preconditioner**
: A transformation applied to a linear system to make it easier for an iterative solver to converge. A good preconditioner approximates the inverse of the coefficient matrix. Common preconditioners include ILU (Incomplete LU), AMG (Algebraic Multigrid), and Jacobi (diagonal).

**Registration**
: In BORES, the process of adding a custom object (solver, preconditioner, well control, boundary function) to the framework's internal registry so it can be referenced by name and serialized/deserialized correctly. Done using decorator functions like `@solver_func`, `@preconditioner_factory`, `@well_control`.

**Renderer**
: In the visualization system, a class responsible for producing a specific type of plot (line, heatmap, volume, etc.). Renderers are used by the `DataVisualizer` class to generate Plotly figures.

**Serialization**
: The process of converting an object into a storable representation (dictionary, JSON, HDF5, etc.) that can be saved to disk and later reconstructed. In BORES, serialization uses the `dump()` function or `to_dict()` method.

**Timestep**
: The discrete time interval over which the simulator advances the solution from one state to the next. Smaller timesteps are more accurate but more expensive. Adaptive timestep control adjusts the interval automatically based on solution behavior.

**Transmissibility**
: See the petroleum term above. In numerical context, transmissibility is precomputed between each pair of adjacent cells and stored in the coefficient matrix. Modifying transmissibility (via faults or fractures) changes the flow connectivity of the grid.

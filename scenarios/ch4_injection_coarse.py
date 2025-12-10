import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full", app_title="SIM3D")


@app.cell
def _():
    import logging
    from pathlib import Path

    import numpy as np
    import sim3D

    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(threshold=np.inf)  # type: ignore
    sim3D.use_64bit_precision()
    sim3D.image_config(scale=3)

    DEPLETED_MODEL_STATE = (
        Path.cwd() / "scenarios/states/primary_depleted_coarse.pkl.xz"
    )

    def main():
        # Load last model state of primary depletion
        state = sim3D.ModelState.load(filepath=DEPLETED_MODEL_STATE)
        model = state.model
        del state

        minimum_miscibility_pressure = 5600.0
        omega = 0.33

        # Gas injection wells, 3-spot pattern
        injection_clamp = sim3D.InjectionClamp()
        control = sim3D.AdaptiveBHPRateControl(
            target_rate=50000,
            target_phase="gas",
            minimum_bottom_hole_pressure=1500,
            clamp=injection_clamp,
        )
        gas_injector_1 = sim3D.injection_well(
            well_name="GI-1",
            perforating_intervals=[((16, 3, 1), (16, 3, 3))],
            radius=0.3542,  # 8.5 inch wellbore
            control=control,
            injected_fluid=sim3D.InjectedFluid(
                name="Methane",
                phase=sim3D.FluidPhase.GAS,
                specific_gravity=0.65,
                molecular_weight=16.04,
                minimum_miscibility_pressure=minimum_miscibility_pressure,
                todd_longstaff_omega=omega,
                is_miscible=True,
            ),
            is_active=True,
            skin_factor=2.0,
        )

        # Create other gas wells as duplicates
        gas_injector_2 = gas_injector_1.duplicate(
            name="GI-2", perforating_intervals=[((16, 16, 1), (16, 16, 3))]
        )
        injectors = [gas_injector_1, gas_injector_2]

        # Production well
        production_clamp = sim3D.ProductionClamp()
        control = sim3D.MultiPhaseRateControl(
            oil_control=sim3D.AdaptiveBHPRateControl(
                target_rate=-150,
                target_phase="oil",
                minimum_bottom_hole_pressure=1200,
                clamp=production_clamp,
            ),
            gas_control=sim3D.AdaptiveBHPRateControl(
                target_rate=-500,
                target_phase="gas",
                minimum_bottom_hole_pressure=1200,
                clamp=production_clamp,
            ),
            water_control=sim3D.AdaptiveBHPRateControl(
                target_rate=-10,
                target_phase="water",
                minimum_bottom_hole_pressure=1200,
                clamp=production_clamp,
            ),
        )
        producer = sim3D.production_well(
            well_name="P-1",
            perforating_intervals=[((14, 10, 3), (14, 10, 4))],
            radius=0.3542,
            control=control,
            produced_fluids=(
                sim3D.ProducedFluid(
                    name="Oil",
                    phase=sim3D.FluidPhase.OIL,
                    specific_gravity=0.845,
                    molecular_weight=180.0,
                ),
                sim3D.ProducedFluid(
                    name="Gas",
                    phase=sim3D.FluidPhase.GAS,
                    specific_gravity=0.65,
                    molecular_weight=16.04,
                ),
                sim3D.ProducedFluid(
                    name="Water",
                    phase=sim3D.FluidPhase.WATER,
                    specific_gravity=1.05,
                    molecular_weight=18.015,
                ),
            ),
            skin_factor=2.5,
            is_active=False,
        )
        producer.schedule_event(
            sim3D.WellEvent(
                hook=sim3D.well_time_hook(time_step=100),
                action=sim3D.well_update_action(is_active=True),
            )
        )
        producers = [producer]

        wells = sim3D.wells_(injectors=injectors, producers=producers)
        options = sim3D.Options(
            scheme="impes",
            total_time=sim3D.Time(
                days=(sim3D.c.DAYS_PER_YEAR * 5) + 100
            ),  # 5 years + 100 days
            time_step_size=sim3D.Time(hours=30),
            max_time_steps=2000,
            output_frequency=1,
            miscibility_model="todd_longstaff",
            use_pseudo_pressure=True,
        )
        states = sim3D.run(model=model, wells=wells, options=options)
        return list(states)

    return Path, main, sim3D


@app.cell
def _(main):
    states = main()
    return (states,)


@app.cell
def _(Path, sim3D, states):
    sim3D.dump_states(
        states,
        filepath=Path.cwd() / "scenarios/states/ch4_injection_coarse_2.pkl",
        exist_ok=True,
        compression="lzma",
    )
    return


if __name__ == "__main__":
    app.run()

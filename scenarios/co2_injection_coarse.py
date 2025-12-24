import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full", app_title="bores")


@app.cell
def _():
    import logging
    from pathlib import Path
    import numpy as np
    import bores

    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(threshold=np.inf)  # type: ignore
    bores.use_64bit_precision()

    DEPLETED_MODEL_STATE = (
        Path.cwd() / "scenarios/states/primary_depleted_coarse.pkl.xz"
    )

    def main():
        # Load last model state of primary depletion
        state = bores.ModelState.load(filepath=DEPLETED_MODEL_STATE)
        model = state.model
        del state

        minimum_miscibility_pressure = 3250.0
        omega = 0.67

        # Gas injection wells, 5-spot pattern
        injection_clamp = bores.InjectionClamp()
        control = bores.AdaptiveBHPRateControl(
            target_rate=50000,
            target_phase="gas",
            minimum_bottom_hole_pressure=1500,
            clamp=injection_clamp,
        )
        gas_injector_1 = bores.injection_well(
            well_name="GI-1",
            perforating_intervals=[((16, 3, 1), (16, 3, 3))],
            radius=0.3542,  # 8.5 inch wellbore
            control=control,
            injected_fluid=bores.InjectedFluid(
                name="CO2",
                phase=bores.FluidPhase.GAS,
                specific_gravity=0.65,
                molecular_weight=44.0,
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

        # Producer well
        production_clamp = bores.ProductionClamp()
        control = bores.MultiPhaseRateControl(
            oil_control=bores.AdaptiveBHPRateControl(
                target_rate=-150,
                target_phase="oil",
                minimum_bottom_hole_pressure=1200,
                clamp=production_clamp,
            ),
            gas_control=bores.AdaptiveBHPRateControl(
                target_rate=-500,
                target_phase="gas",
                minimum_bottom_hole_pressure=1200,
                clamp=production_clamp,
            ),
            water_control=bores.AdaptiveBHPRateControl(
                target_rate=-10,
                target_phase="water",
                minimum_bottom_hole_pressure=1200,
                clamp=production_clamp,
            ),
        )
        producer = bores.production_well(
            well_name="P-1",
            perforating_intervals=[((14, 10, 3), (14, 10, 4))],
            radius=0.3542,
            control=control,
            produced_fluids=(
                bores.ProducedFluid(
                    name="Oil",
                    phase=bores.FluidPhase.OIL,
                    specific_gravity=0.845,
                    molecular_weight=180.0,
                ),
                bores.ProducedFluid(
                    name="Gas",
                    phase=bores.FluidPhase.GAS,
                    specific_gravity=0.65,
                    molecular_weight=16.04,
                ),
                bores.ProducedFluid(
                    name="Water",
                    phase=bores.FluidPhase.WATER,
                    specific_gravity=1.05,
                    molecular_weight=18.015,
                ),
            ),
            skin_factor=2.5,
            is_active=False,
        )
        producer.schedule_event(
            bores.WellEvent(
                hook=bores.well_time_hook(time_step=100),
                action=bores.well_update_action(is_active=True),
            )
        )
        producers = [producer]

        wells = bores.wells_(injectors=injectors, producers=producers)
        timer = bores.Timer(
            initial_step_size=bores.Time(hours=30.0),
            max_step_size=bores.Time(days=2.0),
            min_step_size=bores.Time(hours=6.0),
            simulation_time=bores.Time(
                days=(bores.c.DAYS_PER_YEAR * 5) + 100
            ),  # 5 years
            max_cfl_number=0.9,
            ramp_up_factor=1.2,
            backoff_factor=0.5,
            aggressive_backoff_factor=0.25,
        )
        config = bores.Config(
            scheme="impes",
            output_frequency=1,
            miscibility_model="todd_longstaff",
            use_pseudo_pressure=True,
        )
        states = bores.run(
            model=model,
            timer=timer,
            wells=wells,
            config=config,
        )
        return list(states)

    return Path, main, bores


@app.cell
def _(main):
    states = main()
    return (states,)


@app.cell
def _(Path, bores, states):
    bores.dump_states(
        states,
        filepath=Path.cwd() / "scenarios/states/co2_injection_coarse_2.pkl",
        exist_ok=True,
        compression="lzma",
    )
    return


if __name__ == "__main__":
    app.run()

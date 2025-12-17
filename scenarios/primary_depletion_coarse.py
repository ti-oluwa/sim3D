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
    bores.image_config(scale=3)

    STABILIZED_MODEL_STATE = Path.cwd() / "scenarios/states/stabilized_coarse.pkl.xz"

    def main():
        state = bores.ModelState.load(filepath=STABILIZED_MODEL_STATE)
        model = state.model
        del state

        # Production well
        clamp = bores.ProductionClamp()
        control = bores.MultiPhaseRateControl(
            oil_control=bores.AdaptiveBHPRateControl(
                target_rate=-100,
                target_phase="oil",
                minimum_bottom_hole_pressure=1500,
                clamp=clamp,
            ),
            gas_control=bores.AdaptiveBHPRateControl(
                target_rate=-500,
                target_phase="gas",
                minimum_bottom_hole_pressure=1500,
                clamp=clamp,
            ),
            water_control=bores.AdaptiveBHPRateControl(
                target_rate=-10,
                target_phase="water",
                minimum_bottom_hole_pressure=1500,
                clamp=clamp,
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
            is_active=True,
        )

        producers = [producer]
        wells = bores.wells_(injectors=None, producers=producers)
        options = bores.Options(
            scheme="implicit",
            total_time=bores.Time(days=bores.c.DAYS_PER_YEAR * 5),  # 5 years
            time_step_size=bores.Time(hours=20),
            max_time_steps=2000,
            output_frequency=1,
            miscibility_model="immiscible",
            use_pseudo_pressure=True,
        )
        states = bores.run(model=model, wells=wells, options=options)
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
        filepath=Path.cwd() / "scenarios/states/primary_depletion_coarse.pkl",
        exist_ok=True,
        compression="lzma",
    )

    # Dump last state for use in EOR in next stages
    depleted_state = states[-1]
    depleted_state.dump(
        filepath=Path.cwd() / "scenarios/states/primary_depleted_coarse.pkl",
        compression="lzma",
        exist_ok=True,
    )
    return


if __name__ == "__main__":
    app.run()

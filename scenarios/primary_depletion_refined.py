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

    STABILIZED_MODEL_STATE = Path.cwd() / "scenarios/states/stabilized_refined.pkl.xz"


    def main():
        state = sim3D.ModelState.load(filepath=STABILIZED_MODEL_STATE)
        model = state.model
        del state

        # Production well
        clamp = sim3D.ProductionClamp()
        control = sim3D.MultiPhaseRateControl(
            oil_control=sim3D.AdaptiveBHPRateControl(
                target_rate=-100,
                target_phase="oil",
                minimum_bottom_hole_pressure=1500,
                clamp=clamp,
            ),
            gas_control=sim3D.AdaptiveBHPRateControl(
                target_rate=-500,
                target_phase="gas",
                minimum_bottom_hole_pressure=1500,
                clamp=clamp,
            ),
            water_control=sim3D.AdaptiveBHPRateControl(
                target_rate=-10,
                target_phase="water",
                minimum_bottom_hole_pressure=1500,
                clamp=clamp,
            ),
        )
        producer = sim3D.production_well(
            well_name="P-1",
            perforating_intervals=[((28, 20, 3), (28, 20, 4))],
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
            is_active=True,
        )

        producers = [producer]
        wells = sim3D.wells_(injectors=None, producers=producers)
        options = sim3D.Options(
            scheme="impes",
            total_time=sim3D.Time(days=sim3D.c.DAYS_PER_YEAR * 5),  # 5 years
            time_step_size=sim3D.Time(hours=30),
            max_time_steps=2000,
            output_frequency=1,
            miscibility_model="immiscible",
            use_pseudo_pressure=True,
        )
        states = sim3D.run(model=model, wells=wells, options=options)
        return list(states)
    return Path, main, np, sim3D


@app.cell
def _(main):
    states = main()
    return (states,)


@app.cell
def _(Path, sim3D, states):
    sim3D.dump_states(
        states,
        filepath=Path.cwd() / "scenarios/states/primary_depletion_refined.pkl",
        exist_ok=True,
        compression="lzma",
    )

    # Dump last state for use in EOR in next stages
    depleted_state = states[-1]
    depleted_state.dump(
        filepath=Path.cwd() / "scenarios/states/primary_depleted_refined.pkl",
        compression="lzma",
        exist_ok=True,
    )
    return


if __name__ == "__main__":
    app.run()

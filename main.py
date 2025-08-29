from flight_analysis.campaign import run_campaign
import ac3airborne

if __name__ == "__main__":
    campaign = "AFLUX"
    platform = "P5"
    meta = ac3airborne.get_flight_segments()
    cat = ac3airborne.get_intake_catalog()

    run_campaign(meta, cat, campaign, platform, kind="high_level", out_dir="outputs")

import numpy as np
import polars as pl

from ...constants import SpecialToken as ST
from ..patterns import MatchAndRevise
from ..utils import apply_vocab_to_multitoken_codes, unify_code_names


class DeathData:
    @staticmethod
    @MatchAndRevise(prefix=[ST.DEATH, ST.DISCHARGE], needs_resorting=True)
    def place_death_before_dc_if_same_time(df: pl.DataFrame) -> pl.DataFrame:
        gb_cols = MatchAndRevise.sort_cols
        idx_col = MatchAndRevise.index_col
        return (
            df.sort(pl.col("code").replace_strict(ST.DEATH, 0, default=1, return_dtype=pl.UInt8))
            .group_by(gb_cols, maintain_order=True)
            .agg(pl.col(idx_col).last(), pl.exclude(gb_cols, idx_col))
            .explode(pl.exclude(gb_cols, idx_col))
            .sort(by=idx_col)
            .select(df.columns)
        )


class DemographicData:
    @staticmethod
    @MatchAndRevise(prefix=ST.ADMISSION)
    def retrieve_demographics_from_hosp_adm(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            code=pl.concat_list("code", pl.lit("MARITAL_STATUS"), pl.lit("RACE")),
            text_value=pl.concat_list("text_value", pl.col("marital_status"), pl.col("race")),
        ).explode("code", "text_value")

    @staticmethod
    @MatchAndRevise(prefix="RACE", apply_vocab=True)
    def process_race(df: pl.DataFrame) -> pl.DataFrame:
        race_unknown = ["UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"]
        race_minor = [
            "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER",
            "AMERICAN INDIAN/ALASKA NATIVE",
            "MULTIPLE RACE/ETHNICITY",
        ]
        # every patient can have only one race assigned, so we can prioritize which one to keep
        race_priority_mapping = {"RACE//OTHER": 1, "RACE//UNKNOWN": 2}  # every other will get 0
        return (
            df.with_columns(
                code=pl.when(pl.col("text_value").is_in(race_unknown))
                .then(pl.lit("UNKNOWN"))
                .when(pl.col("text_value").is_in(race_minor))
                .then(pl.lit("OTHER"))
                .when(pl.col("text_value") == "SOUTH AMERICAN")
                .then(pl.lit("HISPANIC"))
                .when(pl.col("text_value") == "PORTUGUESE")
                .then(pl.lit("WHITE"))
                .when(pl.col("text_value").str.contains_any(["/", " "]))
                .then(pl.lit(None))
                .otherwise("text_value")
            )
            .with_columns(
                code=(
                    pl.lit("RACE//")
                    + pl.when(pl.col("code").is_null())
                    .then(pl.col("text_value").str.slice(0, pl.col("text_value").str.find("/| ")))
                    .otherwise("code")
                )
            )
            .group_by(MatchAndRevise.sort_cols[0], maintain_order=True)
            .agg(
                pl.col("code")
                .sort_by(
                    pl.col.code.replace_strict(
                        race_priority_mapping, default=0, return_dtype=pl.UInt8
                    )
                )
                .first(),
                pl.exclude("code").first(),
            )
            .select(df.columns)
        )

    @staticmethod
    @MatchAndRevise(prefix="MARITAL_STATUS", apply_vocab=True)
    def process_marital_status(df: pl.DataFrame) -> pl.DataFrame:
        return df.drop_nulls("text_value").with_columns(
            code=pl.lit("MARITAL//") + pl.col("text_value")
        )


class InpatientData:
    @staticmethod
    @MatchAndRevise(prefix="DRG", apply_vocab=True)
    def process_drg_codes(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col.code.str.starts_with("DRG//HCFA")).with_columns(
            code=pl.lit("DRG//") + pl.col.code.str.split("//").list[2].cast(int).cast(str)
        )

    @staticmethod
    @MatchAndRevise(prefix=ST.ADMISSION)
    def process_hospital_admissions(df: pl.DataFrame) -> pl.DataFrame:
        scheduled_admissions = ["ELECTIVE", "SURGICAL SAME DAY ADMISSION"]
        return (
            df.with_columns(
                pl.col.code.str.split("//").list[0].alias("code"),
                pl.col.code.str.split("//").list[1].alias("text_value"),
            )
            .with_columns(
                pl.concat_list(
                    "code",
                    pl.lit("ADMISSION_TYPE//")
                    + pl.when(
                        pl.col("text_value").str.ends_with("EMER.")
                        | (pl.col("text_value") == "URGENT")
                    )
                    .then(pl.lit("EMERGENCY"))
                    .when(pl.col("text_value").is_in(scheduled_admissions))
                    .then(pl.lit("SCHEDULED"))
                    .otherwise(pl.lit("OBSERVATION")),
                    pl.lit("INSURANCE//") + pl.col("insurance"),
                ).alias("code")
            )
            .explode("code")
        )

    @staticmethod
    @MatchAndRevise(prefix=[ST.DISCHARGE, "DIAGNOSIS//ICD//", "DRG//"])
    def process_hospital_discharges(df: pl.DataFrame) -> pl.DataFrame:
        """Currently must be run before processing diagnoses."""
        discharge_facilities = [
            "HEALTHCARE FACILITY",
            "SKILLED NURSING FACILITY",
            "REHAB",
            "CHRONIC/LONG TERM ACUTE CARE",
            "OTHER FACILITY",
        ]

        drg_following_diag = pl.col.code.str.starts_with(
            "DIAGNOSIS//ICD"
        ) & ~pl.col.code.str.starts_with("DRG//").shift(-1, fill_value=False)
        drg_following_disch = pl.col.code.str.starts_with(ST.DISCHARGE)

        if "stay_id" in df.columns:
            # This means that it is MIMIC with ED extension, and diagnoses in addition come from
            # ED_OUT and in those situations DRG code should not be added
            drg_following_diag &= pl.col.stay_id.is_null() & ~(
                pl.col.code.str.starts_with("DIAGNOSIS//ICD") & pl.col.stay_id.is_null()
            ).shift(-1, fill_value=False)

            drg_following_disch &= pl.col.code.str.starts_with(ST.DISCHARGE).shift(
                -1, fill_value=True
            ) | pl.col.stay_id.is_not_null().shift(-1, fill_value=True)
        else:
            drg_following_diag &= ~pl.col.code.str.starts_with("DIAGNOSIS//ICD").shift(
                -1, fill_value=False
            )
            drg_following_disch &= pl.col.code.str.starts_with(ST.DISCHARGE).shift(
                -1, fill_value=True
            )

        drg_missing_cond = drg_following_diag | drg_following_disch

        return (
            df.with_columns(
                text_value=pl.when(pl.col.code.str.starts_with(ST.DISCHARGE))
                .then(pl.col.code.str.split("//").list[1])
                .otherwise("text_value")
            )
            .with_columns(
                code=pl.when(pl.col.code.str.starts_with(ST.DISCHARGE))
                .then(
                    pl.concat_list(
                        pl.lit(ST.DISCHARGE),
                        (
                            pl.lit("DISCHARGE_LOCATION//")
                            + pl.when(pl.col("text_value").is_in(discharge_facilities))
                            .then(pl.lit("HEALTHCARE_FACILITY"))
                            .when(pl.col("text_value").is_null())
                            .then(pl.lit("UNKNOWN"))
                            .otherwise(pl.col("text_value").replace(" ", "_"))
                        ),
                    )
                )
                .otherwise(pl.concat_list("code")),
                drg_missing=drg_missing_cond,
            )
            .with_columns(
                code=pl.when("drg_missing")
                .then(pl.concat_list("code", pl.lit("DRG//UNKNOWN")))
                .otherwise("code")
            )
            .drop("drg_missing")
            .explode("code")
        )


class MeasurementData:
    @staticmethod
    @MatchAndRevise(prefix=["TEMPERATURE", "HEART_RATE", "RESPIRATORY_RATE", "O2_SATURATION"])
    def process_simple_measurements(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.filter(pl.col("numeric_value").is_not_null())
            .with_columns(
                code=pl.concat_list(
                    pl.lit("VITAL//") + pl.col("code"), pl.lit("VITAL//Q//") + pl.col("code")
                )
            )
            .explode("code")
        )

    @staticmethod
    @MatchAndRevise(prefix="PAIN")
    def process_pain(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.filter(pl.col.text_value.is_not_null())
            .with_columns(
                pl.col("text_value")
                .str.to_lowercase()
                .str.strip_chars(' +"')
                .str.strip_suffix("/10")
            )
            .with_columns(
                numeric_value=pl.when(pl.col.text_value.str.contains("-", literal=True))
                .then(
                    pl.col.text_value.str.split("-").list.first().str.strip_chars() + pl.lit(".5")
                )
                .when(pl.col.text_value.str.contains("crit|moaning"))
                .then(pl.lit("10"))
                .when(pl.col.text_value.str.contains("lot|all over|hurts|much"))
                .then(pl.lit("8"))
                .when(
                    pl.col.text_value.is_in(
                        ["yes", "mild", "moderate", "y", "pain", "uncomfortable"]
                    )
                )
                .then(pl.lit("5"))
                .when(pl.col.text_value.str.contains("little|not bad|some|discomfort"))
                .then(pl.lit("2"))
                .when(
                    pl.col.text_value.str.contains(r"s[lep]{3,4}|sedat|n\/a|resting")
                    | pl.col.text_value.is_in(["no", "no pain", "ok", "none", "comfortable"])
                )
                .then(pl.lit("0"))
                .otherwise(pl.col("text_value").str.replace_all(r"\D", ""))
                .cast(float, strict=False)
            )
            .filter(pl.col.numeric_value.is_between(0, 10))
            .with_columns(code=pl.concat_list(pl.lit("VITAL//PAIN"), pl.lit("VITAL//Q//PAIN")))
            .explode("code")
        )

    @staticmethod
    @MatchAndRevise(prefix="Blood Pressure")
    def process_blood_pressure(bp_df: pl.DataFrame) -> pl.DataFrame:
        return (
            bp_df.with_columns(
                code=pl.when(pl.col.numeric_value.is_null()).then(
                    pl.col("text_value").str.split_exact("/", 1)
                )
                # Hacky way to get the systolic and diastolic that come from ED extension
                .otherwise(
                    pl.struct(
                        field_0=pl.col.numeric_value.cast(int).cast(str),
                        field_1="text_value",
                    )
                )
            )
            .with_columns(
                code=pl.concat_list(
                    pl.lit("VITAL//BLOOD_PRESSURE"),
                    pl.lit("VITAL//Q//SBP"),
                    pl.lit("VITAL//Q//DBP"),
                ),
                numeric_value=pl.concat_list(
                    pl.lit(None),
                    pl.col("code").struct[0].cast(float),
                    pl.col("code").struct[1].cast(float),
                ),
                text_value=pl.col.code.struct[0].cast(str)
                + pl.lit("/")
                + pl.col.code.struct[1].cast(str),
            )
            .explode("code", "numeric_value")
        )


class DiagnosesData:
    @staticmethod
    @MatchAndRevise(prefix="DIAGNOSIS//ICD//")
    def prepare_codes_for_processing(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col.code.str.split_exact("//", 3)).with_columns(
            code=pl.lit("ICD//CM//") + pl.col.code.struct[2], text_value=pl.col.code.struct[3]
        )

    @staticmethod
    @MatchAndRevise(prefix="ICD//CM//9")
    def convert_icd_9_to_10(icd9_df: pl.DataFrame) -> pl.DataFrame:
        from ..mappings import get_icd_cm_9_to_10_mapping

        icd_9_to_10 = get_icd_cm_9_to_10_mapping()
        return (
            icd9_df.with_columns(
                pl.lit("ICD//CM//10").alias("code"),
                pl.col("text_value").replace_strict(icd_9_to_10, default=None),
            )
        ).drop_nulls("text_value")

    @staticmethod
    @MatchAndRevise(prefix="ICD//CM//10", needs_vocab=True)
    def process_icd10(icd10_df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        from ..mappings import get_icd_cm_code_to_name_mapping

        code_to_name = get_icd_cm_code_to_name_mapping()
        temp_cols = ["part1", "part2", "part3"]
        code_prefixes = ["", "3-6//", "SFX//"]
        code_slices = [(0, 3), (3, 3), (6,)]

        df = (
            icd10_df.with_columns(
                pl.col("text_value").str.slice(*code_slice).alias(col)
                for col, code_slice in zip(temp_cols, code_slices)
            )
            .with_columns(pl.col(temp_cols[0]).replace_strict(code_to_name, default=None))
            .with_columns(
                pl.when(pl.col(col) != "")
                .then(pl.lit(f"ICD//CM//{prefix}") + pl.col(col))
                .alias(col)
                for col, prefix in zip(temp_cols, code_prefixes)
            )
            .with_columns(unify_code_names(pl.col(temp_cols)))
        )

        if vocab is not None:
            df = apply_vocab_to_multitoken_codes(df, temp_cols, vocab)

        return (
            df.with_columns(code=pl.concat_list(temp_cols))
            .drop(temp_cols)
            .explode("code")
            .drop_nulls("code")
        )


class ProcedureData:
    @staticmethod
    @MatchAndRevise(prefix="PROCEDURE", needs_resorting=True)
    def prepare_codes_for_processing(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.with_columns(pl.col.code.str.split("//"))
            .filter(pl.col.code.list[1] == "ICD")
            .with_columns(
                code=pl.lit("ICD//PCS//") + pl.col.code.list[2],
                text_value=pl.col.code.list[3],
                time=pl.col("time").dt.combine(time=pl.time(23, 59, 59)),
            )
        )

    @staticmethod
    @MatchAndRevise(prefix="ICD//PCS//9")
    def convert_icd_9_to_10(icd9_df: pl.DataFrame) -> pl.DataFrame:
        from ..mappings import get_icd_pcs_9_to_10_mapping

        icd_9_to_10 = get_icd_pcs_9_to_10_mapping()
        return (
            icd9_df.with_columns(
                pl.lit("ICD//PCS//10").alias("code"),
                pl.col("text_value").replace(icd_9_to_10, default=None),
            )
        ).drop_nulls("text_value")

    @staticmethod
    @MatchAndRevise(prefix="ICD//PCS//10", needs_vocab=True)
    def process_icd10(icd10_df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        df = icd10_df.with_columns(
            pl.col("text_value").str.split_exact("", 6).alias("code")
        ).with_columns(
            code=pl.concat_list(
                pl.when(pl.col("code").struct[i] != "").then(
                    pl.lit("ICD//PCS//") + pl.col("code").struct[i]
                )
                for i in range(7)
            ).list.drop_nulls()
        )
        if vocab is not None:
            # all characters have to be in the vocab to keep the code
            df = df.filter(pl.col("code").list.eval(pl.element().is_in(vocab)).list.all())
        return df.explode("code").drop_nulls("code")


class MedicationData:
    @staticmethod
    @MatchAndRevise(prefix="MEDICATION", needs_vocab=True)
    def convert_to_atc(df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        from ..mappings import get_atc_code_to_desc, get_mimic_drug_name_to_atc_mapping

        drug_to_atc = get_mimic_drug_name_to_atc_mapping()
        code_to_desc = get_atc_code_to_desc()
        temp_cols = ["pfx", "4", "sfx"]
        code_prefixes = ["ATC//", "ATC//4//", "ATC//SFX//"]
        code_slices = [(0, 3), (3, 1), (4,)]

        df = (
            df.with_columns(pl.col("code").str.split("//"))
            .with_columns(
                pl.when(pl.col("code").list[2] == "Administered")
                .then(None)
                .when(pl.col("code").list[1] == "START")
                .then(pl.lit("MEDICATION_START"))
                .alias("code"),
                pl.when(pl.col("code").list[2] == "Administered")
                .then(pl.col("code").list[1])
                .when(pl.col("code").list[1] == "START")
                .then(pl.col("code").list[2])
                .alias("text_value"),
            )
            .drop_nulls("text_value")
            .with_columns(
                pl.col("text_value")
                .str.strip_chars(" ")
                .str.to_lowercase()
                .replace_strict(drug_to_atc, default=None, return_dtype=pl.List(pl.String))
            )
            .with_columns(
                pl.concat_list(
                    "code",
                    pl.lit(None).cast(str).repeat_by(pl.col("text_value").list.len().cast(int) - 1),
                ).alias("code")
            )
            .drop_nulls("text_value")
            .explode("code", "text_value")
            .with_columns(
                pl.col("text_value").str.slice(*slice).alias(col)
                for col, slice in zip(temp_cols, code_slices)
            )
            .with_columns(
                pl.when(pl.col(col) != "")
                .then(
                    pl.lit(pfx)
                    + pl.col(col)
                    + (
                        pl.lit("//") + pl.col(col).replace_strict(code_to_desc, default=None)
                        if pfx == code_prefixes[0]
                        else pl.lit("")
                    )
                )
                .alias(col)
                for col, pfx in zip(temp_cols, code_prefixes)
            )
            .with_columns(unify_code_names(pl.col(temp_cols)))
        )
        if vocab is not None:
            df = apply_vocab_to_multitoken_codes(df, temp_cols, vocab)

        return (
            df.with_columns(code=pl.concat_list("code", *temp_cols))
            .drop(temp_cols)
            .explode("code")
            .drop_nulls("code")
        )


class ICUStayData:
    @staticmethod
    @MatchAndRevise(prefix="ICU_")
    def process(df: pl.DataFrame, *, num_quantiles: int = 10) -> pl.DataFrame:
        from ..mappings import get_stay_id_to_sofa_mapping

        stay_id_to_sofa = get_stay_id_to_sofa_mapping()
        min_value, max_value = min(stay_id_to_sofa.values()), max(stay_id_to_sofa.values())
        bins = np.linspace(min_value, max_value, num_quantiles + 1)
        values = [
            np.arange(np.ceil(left), np.floor(right) + 1)
            for left, right in zip(bins[:-1], bins[1:])
        ]

        # these are not real quantiles, the values are divided equidistantly
        value_to_quantile = {
            value: f"Q{i}" for i, values in enumerate(values, 1) for value in values
        }

        stay_id_to_sofa = {
            stay_id: value_to_quantile[sofa] for stay_id, sofa in stay_id_to_sofa.items()
        }
        return (
            df.with_columns(pl.col("code").str.split("//"))
            .with_columns(
                code=pl.col("code").list[0],
                text_value=pl.lit("ICU_TYPE//") + pl.col("code").list[1],
                sofa_quantiles=pl.col("icustay_id").replace_strict(stay_id_to_sofa, default=None),
            )
            .with_columns(
                pl.when(code=ST.ICU_ADMISSION)
                .then(
                    pl.concat_list(
                        "code",
                        "text_value",
                        pl.when(pl.col("sofa_quantiles").is_not_null())
                        .then(pl.concat_list(pl.lit(ST.SOFA), "sofa_quantiles"))
                        .otherwise([]),
                    ).alias("code")
                )
                .otherwise(pl.concat_list("code"))
            )
            .drop("sofa_quantiles")
            .explode("code")
        )


class TransferData:
    @staticmethod
    @MatchAndRevise(prefix="TRANSFER_TO", apply_vocab=True)
    def retain_only_transfer_and_admit_types(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.with_columns(pl.col.code.str.split("//"))
            .filter(pl.col.code.list[1].is_in(["transfer", "admit"]))
            .with_columns(code=pl.lit("TRANSFER//") + pl.col.code.list[2].fill_null("UNKNOWN"))
        )


class BMIData:
    @staticmethod
    @MatchAndRevise(prefix="BMI")
    def make_quantiles(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.with_columns(
                pl.col("text_value").cast(str).cast(float).alias("numeric_value"),
                pl.lit(None).alias("text_value"),
            )
            .filter(pl.col("numeric_value").is_between(10, 100))
            .with_columns(pl.concat_list(pl.lit("BMI"), pl.lit("BMI//Q")).alias("code"))
            .explode("code")
        )

    @staticmethod
    @MatchAndRevise(prefix=["BMI", "Q"])
    def join_token_and_quantile(df: pl.DataFrame) -> pl.DataFrame:
        """TODO: This one should be done in transform_to_quantiles"""
        q_following_bmi_mask = (pl.col("code") == "BMI").shift(1)
        return df.with_columns(
            code=pl.when(q_following_bmi_mask)
            .then(pl.lit("BMI//") + pl.col("code"))
            .when(pl.col("code") == "BMI")
            .then(None)
            .otherwise("code")
        ).drop_nulls("code")


class LabData:
    @staticmethod
    @MatchAndRevise(prefix="LAB//", apply_vocab=True)
    def retain_only_test_with_numeric_result(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("numeric_value").is_not_null())

    @staticmethod
    @MatchAndRevise(prefix="LAB//", needs_counts=True, needs_vocab=True)
    def make_quantiles(
        df: pl.DataFrame, counts: dict[str, int] | None = None, vocab: list[str] | None = None
    ) -> pl.DataFrame:
        # TODO: we run some simple analysis and decided to keep 200 most frequent labs
        known_lab_names = list(counts.keys())[:200] if vocab is None else vocab
        return (
            df.filter(unify_code_names(pl.col("code")).is_in(known_lab_names))
            .with_columns(pl.concat_list("code", pl.lit("LAB//Q//") + pl.col("code").str.slice(5)))
            .explode("code")
        )


class HCPCSData:
    @staticmethod
    @MatchAndRevise(prefix="HCPCS//", apply_vocab=True)
    def unify_names(df: pl.DataFrame) -> pl.DataFrame:
        """This will just unify the code names."""
        return df


class PatientFluidOutputData:
    @staticmethod
    @MatchAndRevise(prefix="SUBJECT_FLUID_OUTPUT//", needs_vocab=True)
    def make_quantiles(df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        if vocab is not None:
            df.filter(pl.col("code").is_in(vocab))

        prefix = "SUBJECT_FLUID_OUTPUT//"
        return (
            df.filter(pl.col("numeric_value").is_not_null())
            .with_columns(
                pl.concat_list(
                    "code", pl.lit(prefix + "Q//") + pl.col("code").str.slice(len(prefix))
                )
            )
            .explode("code")
        )


class EdData:
    @staticmethod
    @MatchAndRevise(prefix="ED_REGISTRATION")
    def process_ed_registration(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            code=pl.concat_list(
                "code",
                pl.lit("ED_TRANSPORT//")
                + pl.when(pl.col.text_value == "HELICOPTER")
                .then(pl.lit("OTHER"))
                .otherwise("text_value"),
            )
        ).explode("code")

    @staticmethod
    @MatchAndRevise(prefix="ACUITY")
    def process_ed_acuity(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.filter(pl.col("numeric_value").is_not_null())
            .with_columns(
                code=pl.concat_list(
                    "code", pl.lit("Q") + pl.col("numeric_value").cast(pl.UInt8).cast(pl.Utf8)
                )
            )
            .explode("code")
        )

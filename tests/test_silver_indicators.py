import unittest
from datetime import date
from decimal import Decimal

from urban_workflow.silver.indicators import (
    classify_indicator_geography,
    normalize_indicator_barrio_code,
    normalize_indicator_district_code,
    parse_indicator_date,
    parse_indicator_decimal,
)


class SilverIndicatorParsingTests(unittest.TestCase):
    def test_classifies_nullable_geographic_codes(self) -> None:
        self.assertEqual(classify_indicator_geography(None, None), "city")
        self.assertEqual(classify_indicator_geography("7", None), "district")
        self.assertEqual(classify_indicator_geography("7", "75"), "barrio")

    def test_parses_date_and_spanish_decimal(self) -> None:
        self.assertEqual(parse_indicator_date("01/01/2023"), date(2023, 1, 1))
        self.assertEqual(parse_indicator_decimal("1.234,567890123"), Decimal("1234.567890123"))

    def test_normalizes_codes_by_geographic_level(self) -> None:
        self.assertIsNone(normalize_indicator_district_code(None))
        self.assertEqual(normalize_indicator_district_code("7"), "07")
        self.assertIsNone(normalize_indicator_barrio_code(None, "district"))
        self.assertEqual(normalize_indicator_barrio_code("75", "barrio"), "075")

    def test_rejects_invalid_date_decimal_and_codes(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid indicator date"):
            parse_indicator_date("2023-01-01")
        with self.assertRaisesRegex(ValueError, "Invalid indicator numeric value"):
            parse_indicator_decimal("not-a-number")
        with self.assertRaisesRegex(ValueError, "Invalid Madrid district code"):
            normalize_indicator_district_code("22")
        with self.assertRaisesRegex(ValueError, "Invalid Madrid barrio code"):
            normalize_indicator_barrio_code(None, "barrio")


if __name__ == "__main__":
    unittest.main()

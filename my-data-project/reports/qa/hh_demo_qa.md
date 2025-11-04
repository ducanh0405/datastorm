# QA Report: hh_demo
_Generated: 2025-10-29 14:44:00_

## Basic Stats
- Row count: 801
- Column count: 8

## Schema
- household_key: BIGINT
- age_desc: VARCHAR
- marital_status_code: VARCHAR
- income_desc: VARCHAR
- homeowner_desc: VARCHAR
- hh_comp_desc: VARCHAR
- household_size_desc: VARCHAR
- kid_category_desc: VARCHAR

## NULL Percentages
- household_key: 0.00%
- age_desc: 0.00%
- marital_status_code: 0.00%
- income_desc: 0.00%
- homeowner_desc: 0.00%
- hh_comp_desc: 0.00%
- household_size_desc: 0.00%
- kid_category_desc: 0.00%

## Sample Rows
```
   household_key age_desc marital_status_code income_desc homeowner_desc      hh_comp_desc household_size_desc kid_category_desc
0              1      65+                   A      35-49K      HOMEOWNER  2 ADULTS NO KIDS                   2      NONE/UNKNOWN
1              7    45-54                   A      50-74K      HOMEOWNER  2 ADULTS NO KIDS                   2      NONE/UNKNOWN
2              8    25-34                   U      25-34K        UNKNOWN     2 ADULTS KIDS                   3                 1
3             13    25-34                   U      75-99K      HOMEOWNER     2 ADULTS KIDS                   4                 2
4             16    45-54                   B      50-74K      HOMEOWNER     SINGLE FEMALE                   1      NONE/UNKNOWN
```

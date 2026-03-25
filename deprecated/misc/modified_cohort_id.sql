-- Written by Zewei “Whiskey” Liao @whiskey0504
-- Cf. https://raw.githubusercontent.com/Common-Longitudinal-ICU-data-Format/CLIF-epi-of-sedation/c75107d0738dd75db0e6b81789ad8ef3c0d1f5f6/code/cohort_id.sql


WITH t1 AS (
    FROM resp_p
    SELECT hospitalization_id
        , event_dttm: recorded_dttm
        , device_category
        , _on_imv: CASE WHEN device_category = 'imv' THEN 1 ELSE 0 END
        , _chg_imv: CASE
            -- getting off imv (extub)
            WHEN (_on_imv = 0 AND LAG(_on_imv) OVER w = 1)
            -- getting on imv (intub)
            OR (_on_imv = 1 AND _on_imv IS DISTINCT FROM LAG(_on_imv) OVER w)
            THEN 1 ELSE 0 END
        , _trach_flip_to_1: CASE
            WHEN tracheostomy = 1 AND tracheostomy IS DISTINCT FROM LAG(tracheostomy) OVER w THEN 1 ELSE 0 END
    WINDOW w AS (PARTITION BY hospitalization_id ORDER BY event_dttm)
)
, t2 as (
    FROM t1
    SELECT *
        , _streak_id: SUM(_chg_imv) OVER w
        , _trach_flip_cumsum: SUM(_trach_flip_to_1) OVER w
        , _trach_1st: CASE
            WHEN _trach_flip_to_1 = 1 AND _trach_flip_cumsum = 1 THEN 1 ELSE 0 END
    WINDOW w AS (PARTITION BY hospitalization_id ORDER BY event_dttm)
)
, all_streaks as (
    FROM t2
    SELECT hospitalization_id
        , _streak_id
        , _start_dttm: MIN(event_dttm)
        , _last_observed_dttm: MAX(event_dttm)
        , _trach_dttm: MIN(CASE WHEN _trach_1st = 1 THEN event_dttm END)
        , _on_imv: MAX(_on_imv)
    -- WHERE _on_imv = 1
    GROUP BY hospitalization_id, _streak_id
)
, all_streaks_w_lead as (
    FROM all_streaks
    SELECT *
        -- the end time of the current streak is the start time of the next streak
        , _next_start_dttm: LEAD(_start_dttm) OVER w
        , _end_dttm: COALESCE(_trach_dttm, _next_start_dttm, _last_observed_dttm)
        , _duration_hrs: date_diff('minute', _start_dttm, _end_dttm) / 60
        , _at_least_24h: CASE WHEN _duration_hrs >= 24 THEN 1 ELSE 0 END
    WINDOW w AS (PARTITION BY hospitalization_id ORDER BY _streak_id)
)

--, hosp_included as ( -- keep only hospitalizations that have at least one IMV streak of 24 hours or longer
--    FROM all_streaks_w_lead
--    SELECT hospitalization_id
--        , MAX(_at_least_24h) as _on_imv_24h_or_longer
--    GROUP BY hospitalization_id
--    HAVING _on_imv_24h_or_longer = 1
--)
, t3 as (
    FROM t2
    LEFT JOIN all_streaks_w_lead USING (hospitalization_id, _streak_id)
    SELECT t2.*
        , _duration_hrs
--    WHERE hospitalization_id IN (
--        SELECT hospitalization_id
--        FROM hosp_included
-- ) AND _streak_id >= 1 -- remove rows before the first imv streak
)

SELECT *
FROM all_streaks_w_lead
-- ORDER BY hospitalization_id, event_dttm
ORDER BY hospitalization_id, _streak_id

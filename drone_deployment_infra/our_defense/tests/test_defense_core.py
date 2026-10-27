"""Strategy: assignment, ramming aim, goalkeeper, formations, flight box."""

import numpy as np
import pytest

from defense_core import BrainCfg, DefenseBrain, Geofence, Track

FENCE = Geofence(np.array([-3.0, -0.5, 0.4]), np.array([3.0, 0.5, 6.0]))


def brain(**kw) -> DefenseBrain:
    return DefenseBrain(BrainCfg(site=np.array([0.0, 0.0, 4.0]), fence=FENCE,
                                 front=np.array([0.0, 0.0, -1.0]), **kw))


def test_every_attacker_gets_a_defender_and_the_spare_keeps_goal():
    b = brain()
    defenders = {"D1": np.array([-1.0, 0, 3.0]), "D2": np.array([0.0, 0, 3.0]), "D3": np.array([1.0, 0, 3.0])}
    attackers = [Track("A1", np.array([-1.0, 0, 1.0])), Track("A2", np.array([1.2, 0, 1.0]))]
    plans = b.plan(defenders, attackers)
    rammed = sorted(p.attacker for p in plans.values() if p.mode == "ram")
    assert rammed == ["A1", "A2"]
    assert [p.mode for p in plans.values()].count("guard") == 1
    # nearest pairing: D1 takes A1 (left), D3 takes A2 (right), D2 (middle) keeps goal
    assert plans["D1"].attacker == "A1" and plans["D3"].attacker == "A2" and plans["D2"].mode == "guard"


def test_downed_or_stale_attackers_are_ignored():
    b = brain(formation="line")
    defenders = {"D1": np.array([0.0, 0, 3.0])}
    plans = b.plan(defenders, [Track("A1", np.array([0.0, 0, 1.0]), alive=False),
                               Track("A2", np.array([1.0, 0, 1.0]), age=5.0)])
    assert plans["D1"].mode == "hold"


def test_default_aim_is_the_attackers_current_position():
    b = brain()
    a = Track("A", np.array([1.0, 0.0, 2.0]), vel=np.array([0.3, 0.0, 0.2]), age=0.1)
    meet, _ = b.intercept(np.array([0.0, 0.0, 3.0]), a)
    # only the 0.1 s report age is brought up to now
    assert np.allclose(meet, a.pos + a.vel * 0.1, atol=1e-6)


def test_lead_mode_meets_the_attacker_at_the_same_time():
    b = brain(defender_speed=0.5, max_lead_s=3.0)
    d = np.array([0.0, 0.0, 1.5])
    a = Track("A", np.array([2.0, 0.0, 3.0]), vel=np.array([-0.3, 0.0, -0.2]), age=0.0)
    meet, lead = b.intercept(d, a)
    assert np.linalg.norm(meet - d) / 0.5 == pytest.approx(lead, abs=1e-6)       # we fly there in `lead` s
    assert np.allclose(meet, a.pos + a.vel * lead, atol=1e-6)                     # and so does it


def test_v_formation_tip_toward_tag10_and_no_crossing():
    b = brain(formation="v", v_center=np.array([1.0, 0.0, 3.0]), v_width=1.2, v_depth=0.6)
    pos = {"L": np.array([0.2, 0, 4.0]), "M": np.array([1.0, 0, 4.0]), "R": np.array([1.9, 0, 4.0])}
    slots = b.hold_points(list(pos), pos)
    assert np.allclose(slots["M"][[0, 2]], [1.0, 3.0])                # middle defender takes the tip
    assert np.allclose(slots["L"][[0, 2]], [0.4, 3.6])                # arms trail away from tag 10
    assert np.allclose(slots["R"][[0, 2]], [1.6, 3.6])


def test_home_formation_returns_each_defender_to_its_start():
    b = brain(formation="home")
    b.anchors = {"D1": np.array([0.5, 0.1, 2.5])}
    plans = b.plan({"D1": np.array([2.0, 0.0, 1.0])}, [])
    assert plans["D1"].mode == "hold" and np.allclose(plans["D1"].target, [0.5, 0.1, 2.5])


def test_targets_are_clamped_to_the_flight_box():
    b = brain()
    plans = b.plan({"D1": np.array([0.0, 0, 3.0])}, [Track("A", np.array([9.0, 2.0, -4.0]))])
    t = plans["D1"].target
    assert np.all(t >= FENCE.lo - 1e-9) and np.all(t <= FENCE.hi + 1e-9)

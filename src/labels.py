"""The six appeal reasons and the definitions the LLM is given.

Clear, mutually exclusive definitions matter more than prompt tricks:
most early mistakes came from overlapping categories, not from the model.
"""

LABELS = {
    "blue_badge": "The driver or a passenger holds a Blue Badge / disabled badge, "
                  "or the appeal is about use of accessible bays.",
    "signage": "The driver says signs were missing, hidden, unclear, unlit, damaged, "
               "contradictory or did not show the terms they were charged under.",
    "payment_error": "The driver paid or tried to pay, but the payment failed, was not "
                     "recorded, went against the wrong vehicle/site, or a machine/app failed.",
    "permit": "The driver holds, or should hold, a permit (resident, staff, student, "
              "visitor, contractor) and the problem is about that permit.",
    "emergency": "A medical emergency, breakdown, accident, evacuation, bereavement or "
                 "similar event outside the driver's control stopped them complying.",
    "other": "Anything else: not the driver, sold the car, cloned plates, grace period, "
             "charge too high, letters not received.",
}

LABEL_NAMES = list(LABELS)

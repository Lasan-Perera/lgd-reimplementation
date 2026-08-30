# Language grounding test 2: fork/knife/cup

Fresh AI-generated image (not in training set), three prompts naming three
different objects, released pretrained checkpoint.

| Prompt | Predicted box |
|---|---|
| "grasp the knife at its handle" | Correctly on the knife handle |
| "grasp the fork" | Same location (on the knife) |
| "grasp the cup" | Same location (on the knife) |

Second independent confirmation of the pattern first seen in the duck/apple
test (see ../README.md). Notably stronger evidence than that test: this image
demonstrates the model CAN localize correctly (knife handle result is
genuinely good), which rules out "broken geometry" as the explanation for
the fork/cup failures. The pattern is specific to text not redirecting
attention once a region is settled on -- consistent with discrepancy #5
(text-visual fusion line commented out, y_flatten never gradiented).

# Language grounding test

Same image, two different grasp instructions, released pretrained checkpoint.

Image: grasp-anything++/image/000018668931a8fb14891fa2b4c0aaa4b50334d17c20d7d2f6306cc47b2f9830.jpg
(contains a rubber duck and a green apple)

| Prompt | Predicted box |
|---|---|
| "Pick up apple by its skin." | On the duck |
| "grasp the duck" | On the duck (near-identical box) |

Both prompts produced essentially the same grasp rectangle — same position,
size, and orientation — despite specifying different objects. This is
consistent with discrepancy #5 in notes/paper_vs_code.md: the text-visual
fusion line (`img = img + y`) is commented out in the released network,
leaving the ALBEF attention mask as the sole language pathway. This single
test suggests that pathway may not meaningfully steer prediction on at
least some images -- the model appears to default to the visually dominant
object in the scene regardless of instruction.

Caveat: n=1 image, one pair of prompts. Not yet systematic. See "Next steps"
in notes/paper_vs_code.md.

## **1. Focal Loss**

Normally, **Cross-Entropy Loss** treats every pixel equally.
That means even **easy pixels** (background that’s clearly not tumor) affect the loss.

But in segmentation, most pixels are easy background, so the model can get lazy and ignore the **hard ones** (like tumor edges).

**Focal Loss** fixes this by:

* Giving **less weight** to pixels the model already gets right (easy ones).
* Giving **more weight** to **hard pixels** the model struggles with.

**Effect:** The model learns better around tricky regions, like tumor boundaries or small details.

---

## **2. Tversky Loss**

Tversky Loss is a modified form of **Dice Loss**.\
Dice treats **false positives** and **false negatives** equally.\
But in medical images, these two errors are **not equally bad**:

* **False Positive (FP):** Saying there’s a tumor when there isn’t → very bad (unnecessary worry or treatment).
* **False Negative (FN):** Missing a real tumor → also bad, but sometimes less common.

Tversky Loss adds two parameters (α and β) to control this trade-off:
$$
\text{Tversky Index} = \frac{TP}{TP + \alpha FP + \beta FN}
$$

* If **α > β**, it **penalizes false positives more**.
* If **β > α**, it **penalizes false negatives more**.

**Effect:** You can tune the model’s sensitivity, in this case, make it more careful not to mark healthy areas as tumor.

> NOTE: choosing whether to penaliz **false positives** or **false negatives** more depends on your specific application and how the model behaves.
>
> In our case, the model predicted a lot of false tumor pixels, so we increased the penalty on false positives to make it more conservative.

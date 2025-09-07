using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ConfineInSquare : MonoBehaviour
{
    //this is for clamping the movement of the attached object with in a box (modifiable)
    //the size of the box is to be modified!!!

    public GameObject confiningBox;
    private Vector3 minBounds;
    private Vector3 maxBounds;
    private BoxCollider boxCollider;

    void Start()
    {
        boxCollider = confiningBox.GetComponent<BoxCollider>();

        if (boxCollider == null)
        {
            Debug.LogError("No BoxCollider found on the ConfiningBox GameObject");
            return;
        }

        UpdateBounds();
    }

    void UpdateBounds()
    {
        minBounds = confiningBox.transform.TransformPoint(boxCollider.center - boxCollider.size / 2);
        maxBounds = confiningBox.transform.TransformPoint(boxCollider.center + boxCollider.size / 2);
    }

    void LateUpdate()
    {
        if (boxCollider != null)
        {
            UpdateBounds();
            Vector3 newPosition = transform.position;
            newPosition.x = Mathf.Clamp(newPosition.x, minBounds.x, maxBounds.x);
            newPosition.y = Mathf.Clamp(newPosition.y, minBounds.y, maxBounds.y);
            newPosition.z = Mathf.Clamp(newPosition.z, minBounds.z, maxBounds.z);
            transform.position = newPosition;
        }
    }
}

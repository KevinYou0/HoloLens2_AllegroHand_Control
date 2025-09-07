using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CollisionDetection : MonoBehaviour
{
    // Public GameObjects for the items in Group 1
    public GameObject itemA;
    public GameObject itemB;
    public GameObject itemC;
    public GameObject itemD;

    // Public GameObject for the item in Group 2
    public GameObject itemX;

    // Public integers to keep track of collision counts
    public int collisionCountA = 0;
    public int collisionCountB = 0;
    public int collisionCountC = 0;
    public int collisionCountD = 0;

    // Update is called once per frame
    void Update()
    {
        CheckCollision(itemA, ref collisionCountA);
        CheckCollision(itemB, ref collisionCountB);
        CheckCollision(itemC, ref collisionCountC);
        CheckCollision(itemD, ref collisionCountD);
    }

    // Check for collision and update count
    void CheckCollision(GameObject group1Item, ref int collisionCount)
    {
        // Check if Group 1 item's collider is intersecting with Group 2 item's collider
        if (group1Item.GetComponent<Collider>().bounds.Intersects(itemX.GetComponent<Collider>().bounds))
        {
            collisionCount++;
        }
    }
}

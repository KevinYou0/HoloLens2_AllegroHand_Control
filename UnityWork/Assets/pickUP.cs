using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class pickUP : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown("q")){
            this.GetComponent<Rigidbody>().isKinematic = false; // make the rigidbody work again

            this.transform.parent = null;
        }
    }

    void OnCollisionEnter(Collision collision)
    {
       
        if(collision.gameObject.tag == "hand")
        {
            this.GetComponent<Rigidbody>().isKinematic = true;
            this.transform.parent = collision.transform;
        }
        Debug.Log(collision.gameObject.name);
    }
}

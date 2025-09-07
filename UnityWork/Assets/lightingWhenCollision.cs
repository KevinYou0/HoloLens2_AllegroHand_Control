using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class lightingWhenCollision : MonoBehaviour
{
    // Start is called before the first frame update
    public int redCol;
    public int greenCol;
    public int blueCol;
    public bool CollisionExsiting = false;
    public bool flashingIn = true;
    public bool startedFlashing = false;
    private Color initialColor; 
    void Start()
    {
        initialColor = this.GetComponent<Renderer>().material.color;
    }

    // Update is called once per frame
    void Update()
    {
        this.GetComponent<Renderer>().material.color = new Color32((byte)redCol, (byte)greenCol, (byte)blueCol, 255);
    }
    private void OnTriggerStay(Collider other)
    {
        Debug.Log(other.gameObject.name);
        if (other.gameObject.tag == "Body")
        {

            CollisionExsiting = true;
            if (startedFlashing == false)
            {
                startedFlashing = true;
                // StartCoroutine(FlashObject());
                this.GetComponent<Renderer>().material.color = new Color32(255, (byte)greenCol, (byte)blueCol, 255);
            }
        }
     
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.gameObject.tag == "Body")
        {
            CollisionExsiting = false;
            startedFlashing = false;
            StopCoroutine(FlashObject());
            //this.GetComponent<Renderer>().material.color = new Color32((byte)redCol, (byte)greenCol, (byte)blueCol, 255);
            redCol = 0;
            greenCol = 0;
            blueCol = 0;
            this.GetComponent<Renderer>().material.color = initialColor;
        }

      
    }

    IEnumerator FlashObject()
    {
        while(CollisionExsiting == true)
        {
            yield return new WaitForSeconds(0.05f);
            if(flashingIn == true)
            {
                if(blueCol <= 30)
                {
                    flashingIn = false;
                }
                else
                {
                    redCol -= 25;
                    greenCol -= 1;
                }
            }
            if (flashingIn == false)
            {
                if (blueCol >= 250)
                {
                    flashingIn = true;
                }
                else
                {
                    redCol += 25;
                    greenCol += 1;
                }
            }
        }
    }
}

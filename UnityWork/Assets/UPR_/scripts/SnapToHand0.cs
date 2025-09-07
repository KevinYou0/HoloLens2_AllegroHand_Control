using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SnapToHand0 : MonoBehaviour
{

    public GameObject CenterCube;
    //items for grabbing (1st stage)
    public List<GameObject> CubesToGrab;

    //items for following hand (2nd stage)
    public List<GameObject> CubesForGrabVirtual;

    public GameObject DataControl, virtualGrabbingPosition, virtualInsersionPosition;

    public GameObject CSVWritterForCondition;
    public GameObject targetHandObject0;
    private string conditionNumber;

    private Vector3 handposition, targetOriginalPos;
    public Material objTograbMaterial;

    private bool grabbed = false, inserted = false;

    //set reference target point for virtual insersion
    public GameObject CubBigRef_Recline, CubBigRef_CylBig, CubBigRef_CylSmall, CubBigRef_Star;

    //String for conditions to data output
    public string ConditionIG = "00";

    //Target Area Sphere
    public GameObject TargetRedCube, TargetAreaGrab, TargetAreaInsertion; // Assign this in the inspector
    public Material YellowMaterial; // Assign in the inspector
    public Material TransparentMaterial; // Assign in the inspector
    private MeshRenderer GrabAreaRenerer, InsertionAreaRenderer;
    public bool isCubeInside = false;
    private float GrabAreaRadius, InsertionAreaRadius;


    void Start()
    {
        foreach (var obj in CubesToGrab)
        {
            //obj.GetComponent<Renderer>().enabled = false;
            obj.SetActive(false);
        }
        foreach (var objV in CubesForGrabVirtual)
        {
            //obj.GetComponent<Renderer>().enabled = false;
            objV.SetActive(false);
        }

        GrabAreaRadius = TargetAreaGrab.GetComponent<SphereCollider>().radius * TargetAreaGrab.transform.localScale.x;
        InsertionAreaRadius = TargetAreaInsertion.GetComponent<SphereCollider>().radius * TargetAreaInsertion.transform.localScale.x;

        GrabAreaRenerer = TargetAreaGrab.GetComponent<MeshRenderer>();
        InsertionAreaRenderer = TargetAreaInsertion.GetComponent<MeshRenderer>();

        GrabAreaRenerer.material = YellowMaterial; // Start with ExpArea as yellow
        InsertionAreaRenderer.material = YellowMaterial; // Start with ExpArea as yellow
        
        TargetAreaGrab.SetActive(false);
        TargetAreaInsertion.SetActive(false);
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.N))    //N for insersion
        {
            conditionNumber = "00";
            inserted = false;
            DisableAllReferenceBigCubes();
            ConditionIG = "00";
            ResetExpAreaInsert();
            isCubeInside = false;
        }
        if (Input.GetKeyDown(KeyCode.M))    //M for gripping
        {
            conditionNumber = "01";
            grabbed = false;
            SetBackVirtualobj();
            ConditionIG = "00";
            ResetExpAreaGrab();
            isCubeInside = false;
        }
        //Debug.Log(conditionNumber);
        //Debug.Log("Gripping Center Distance: " + grippingCenterCompare());
        //conditionNumber = CSVWritterForCondition.GetComponent<CSVWritter1>().ConditionNum;

        CheckCubeStatusGrab();
        CheckCubeStatusInsert();


        if (conditionNumber == "00")    //this is for inserting physical object, virtual obj follows hand
        {
            UpdatePositions();

            if (Input.GetKeyDown("1"))
            { 
                EnableGrabItems(0);
                CubBigRef_Recline.SetActive(true);
                ConditionIG = "I1";
                isCubeInside = false;
            }
            if (Input.GetKeyDown("2")) 
            { 
                EnableGrabItems(1);
                CubBigRef_CylBig.SetActive(true);
                ConditionIG = "I2";
                isCubeInside = false;
            }
            if (Input.GetKeyDown("3")) 
            { 
                EnableGrabItems(2);
                CubBigRef_CylSmall.SetActive(true);
                ConditionIG = "I3";
                isCubeInside = false;
            }
            if (Input.GetKeyDown("4")) 
            { 
                EnableGrabItems(3);
                CubBigRef_Star.SetActive(true);
                ConditionIG = "I4";
                isCubeInside = false;
            }
            if (Input.GetKeyDown("0")) 
            { 
                DisableAllRenderers();
                DisableAllReferenceBigCubes();
            }


            bool ifGrippingCenterSatisfied = false;

            if (InsertingCenterCompareZXPlane() <= 0.006f)  //when centrer of hand is close enough in X-Y plane, change color to blue
            {
                objTograbMaterial.color = Color.blue;
                ifGrippingCenterSatisfied = true;

                //if (ifGrippingCenterSatisfied && targetHandObject0.GetComponent<TargetControl0>().fingerDistance <= 0.06f)
                if (ifGrippingCenterSatisfied && InsertingCenterCompareYAxisPositiveOnly() <= 0.015f)
                {
                    objTograbMaterial.color = Color.green;
                    inserted = true;
                }
            }
            else
            {
                ifGrippingCenterSatisfied = false;
                objTograbMaterial.color = Color.white;
            }

            if (inserted)
            {
                objTograbMaterial.color = Color.green;
                //UpdatePositionsVirtual();
                SetBackInsertionobj();
                ConditionIG = "00";
                isCubeInside = false;
            }

        }
        else if (conditionNumber == "01")   //this is for grabbing virtual object, virtual obj follows hand when gripping is complete
        {
            if (Input.GetKeyDown("1")) 
            { 
                EnableVirtualItems(0);
                ConditionIG = "G1";
                isCubeInside = false;
            }
            if (Input.GetKeyDown("2")) 
            { 
                EnableVirtualItems(1);
                ConditionIG = "G2";
                isCubeInside = false;
            }
            if (Input.GetKeyDown("3")) 
            { 
                EnableVirtualItems(2);
                ConditionIG = "G3";
                isCubeInside = false;
            }
            if (Input.GetKeyDown("4")) 
            { 
                EnableVirtualItems(3);
                ConditionIG = "G4";
                isCubeInside = false;
            }
            if (Input.GetKeyDown("0")) 
            { 
                DisableAllRenderers(); 

            }

            bool isGrippingCenterSatisfied = false;

            if (grippingCenterCompare() <= 0.006f)  //when centrer of hand is close enough, change color to blue
            {
                objTograbMaterial.color = Color.blue;
                isGrippingCenterSatisfied = true;

                if (isGrippingCenterSatisfied && targetHandObject0.GetComponent<TargetControl0>().fingerDistance <= 0.05f)
                {
                    objTograbMaterial.color = Color.green;
                    grabbed = true;
                }
            }
            else
            {
                isGrippingCenterSatisfied = false;
                objTograbMaterial.color = Color.white;
            }

            if (grabbed)
            {
                objTograbMaterial.color = Color.green;
                UpdatePositionsVirtual();
                ConditionIG = "00";
                isCubeInside = false;
            }

        }

    }

    void OnApplicationQuit()
    {
        objTograbMaterial.color = Color.white;
    }


    private float grippingCenterCompare()
    {
        handposition = targetHandObject0.transform.position;

        //this is compared to a fixed position
        targetOriginalPos = virtualGrabbingPosition.transform.position;
        float distance = (handposition - targetOriginalPos).magnitude;
        return distance;
    }

    private float insertingCenterCompare()
    {
        handposition = targetHandObject0.transform.position;

        //this is compared to a fixed position
        targetOriginalPos = virtualInsersionPosition.transform.position;
        float distance = (handposition - targetOriginalPos).magnitude;
        return distance;
    }

    private float InsertingCenterCompareZXPlane()  //new, gpt
    {
        Vector3 handPosition = targetHandObject0.transform.position;
        Vector3 targetOriginalPos = virtualInsersionPosition.transform.position;

        // Setting Y values to 0 to ignore Y-axis in distance calculation
        handPosition.y = 0;
        targetOriginalPos.y = 0;

        float distance = (handPosition - targetOriginalPos).magnitude;
        return distance;
    }

    private float InsertingCenterCompareYAxisPositiveOnly()   //new, gpt
    {
        Vector3 handPosition = targetHandObject0.transform.position;
        Vector3 targetOriginalPos = virtualInsersionPosition.transform.position;

        // Calculate Y-axis distance
        float yDistance = handPosition.y - targetOriginalPos.y;

        // Return 0 if Y distance is negative, else return the positive Y distance
        return yDistance > 0 ? yDistance : -yDistance;
    }


    void EnableGrabItems(int index)
    {
        DisableAllRenderers();

        //CubesToGrab[index].GetComponent<Renderer>().enabled = true;
        //CubesForGrabVirtual[index].GetComponent<Renderer>().enabled = true;

        //change from toggle rendere to toggle active
        CubesToGrab[index].SetActive(true);
        //CubesForGrabVirtual[index].SetActive(true);
    }

    void EnableVirtualItems(int index)
    {
        DisableAllRenderers();

        CubesForGrabVirtual[index].SetActive(true);
    }

    void DisableAllRenderers()
    {
        foreach (var obj in CubesToGrab)
        {
            //obj.GetComponent<Renderer>().enabled = false;
            obj.SetActive(false);
        }

        foreach (var objV in CubesForGrabVirtual)
        {
            //obj.GetComponent<Renderer>().enabled = false;
            objV.SetActive(false);
        }
    }

    void UpdatePositions()
    {
        foreach (var obj in CubesToGrab)
        {
            obj.transform.position = CenterCube.transform.position;

            obj.transform.rotation = CenterCube.transform.rotation*Quaternion.Euler(90,0,0);
        }
    }

    void UpdatePositionsVirtual()
    {
        foreach (var obj in CubesForGrabVirtual)
        {
            obj.transform.position = CenterCube.transform.position;

            obj.transform.rotation = CenterCube.transform.rotation * Quaternion.Euler(90, 0, 0);
        }
    }

    void SetBackVirtualobj()
    {
        foreach (var obj in CubesForGrabVirtual)
        {
            obj.transform.position = virtualGrabbingPosition.transform.position;

            obj.transform.rotation = virtualGrabbingPosition.transform.rotation;// * Quaternion.Euler(90, 0, 0);
        }
    }

    void SetBackInsertionobj()
    {
        foreach (var obj in CubesToGrab)
        {
            obj.transform.position = virtualInsersionPosition.transform.position;

            obj.transform.rotation = virtualInsersionPosition.transform.rotation * Quaternion.Euler(90, 0, 0);
        }
    }

    void DisableAllReferenceBigCubes()
    {
        CubBigRef_Recline.SetActive(false);
        CubBigRef_CylBig.SetActive(false);
        CubBigRef_CylSmall.SetActive(false);
        CubBigRef_Star.SetActive(false);
    }


    void CheckCubeStatusGrab()
    {
        float distance = Vector3.Distance(TargetAreaGrab.transform.position, TargetRedCube.transform.position);
        if (distance <= GrabAreaRadius)
        {
            if (!isCubeInside)
            {
                GrabAreaRenerer.material = TransparentMaterial; // Change to transparent
                isCubeInside = true; // Set flag true when cube enters
            }
        }
        else
        {
            if (isCubeInside)
            {
                //ResetExpAreaGrab(); // Optionally reset when cube exits
            }
        }
    }

    void ResetExpAreaGrab()
    {
        TargetAreaGrab.SetActive(true);
        TargetAreaInsertion.SetActive(false);
        GrabAreaRenerer.material = YellowMaterial; // Reset color to yellow
        isCubeInside = false; // Reset your bool flag
    }

    void CheckCubeStatusInsert()
    {
        float distance = Vector3.Distance(TargetAreaInsertion.transform.position, TargetRedCube.transform.position);
        if (distance <= InsertionAreaRadius)
        {
            if (!isCubeInside)
            {
                InsertionAreaRenderer.material = TransparentMaterial; // Change to transparent
                isCubeInside = true; // Set flag true when cube enters
            }
        }
        else
        {
            if (isCubeInside)
            {
                //ResetExpAreaInsert(); // Optionally reset when cube exits
            }
        }
    }

    void ResetExpAreaInsert()
    {
        TargetAreaGrab.SetActive(false);
        TargetAreaInsertion.SetActive(true);
        InsertionAreaRenderer.material = YellowMaterial; // Reset color to yellow
        isCubeInside = false; // Reset your bool flag
    }
}

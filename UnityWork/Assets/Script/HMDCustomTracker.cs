//using System.Collections;
//using System.Collections.Generic;
//using UnityEngine;
//using ViveSR.anipal.Eye;

//public class HMDCustomTracker : MonoBehaviour
//{
//    // Start is called before the first frame update
//    // for sr
//    public GameObject EyeSR;
//    private VerboseData verbose_Data;
    
//    // for lumin
//    public GameObject Lum;
//    private double lum;

//    void Start()
//    {
//        GetDataFromOtherComponent();
//    }

//    // Update is called once per frame
//    void Update()
//    {
//        GetDataFromOtherComponent();


      
//    }

//    // Get data from outside components 
//    void GetDataFromOtherComponent()
//    {
//        verbose_Data = EyeSR.GetComponent<TestGetData>().verbose_Data;
//        lum = Lum.GetComponent<Luminosity>().lum;
//    }
//}

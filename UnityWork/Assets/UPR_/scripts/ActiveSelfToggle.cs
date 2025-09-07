using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ActiveSelfToggle : MonoBehaviour
{
    [SerializeField]
    private GameObject self;

    private void Start()
    {
        self.SetActive(false);
    }

    public void OnSelf()
    {
        self.SetActive(true);
    }

    public void OffSelf()
    {
        self.SetActive(false);
    }

    public void ToggleSelfActive()
    {
        self.SetActive(!self.activeSelf);
        //Debug.Log("toggled active self");
    }

}

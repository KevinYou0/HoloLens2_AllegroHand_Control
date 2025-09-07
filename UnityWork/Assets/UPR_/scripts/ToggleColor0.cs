using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ToggleColor0 : MonoBehaviour
{
    private Renderer _renderer;
    private bool _isRed;
    
    // Start is called before the first frame update
    void Start()
    {
        _renderer = GetComponent<Renderer>();
        _isRed = true;
        SetColor(Color.red);
    }

    public void ToggleColor() 
    { 
        if(_isRed)
        {
            SetColor(Color.green);
        }
        else 
        {
            SetColor(Color.red);
        }

        _isRed = !_isRed;
    }

    private void SetColor(Color color)
    {
        _renderer.material.color = color;
        Debug.Log("cOLORis SETTTTTTTT");
    }
}

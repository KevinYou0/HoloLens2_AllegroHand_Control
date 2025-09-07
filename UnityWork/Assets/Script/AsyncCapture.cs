using UnityEngine;
using Unity.Collections;
using System.IO;
using System.Collections.Generic;

#if UNITY_2018_3_OR_NEWER
using UnityEngine.Rendering;
#else
using UnityEngine.Experimental.Rendering;
#endif

public class AsyncCapture : MonoBehaviour
{
    Queue<AsyncGPUReadbackRequest> _requests = new Queue<AsyncGPUReadbackRequest>();
    private  Color32[] cArr = new Color32[0];
    public double lumenValue;

    void Update()
    {

        while (_requests.Count > 0)
        {
            var req = _requests.Peek();

            if (req.hasError)
            {
                Debug.Log("GPU readback error detected.");
                _requests.Dequeue();
            }
            else if (req.done)
            {


                var buffer = req.GetData<Color32>();
              
                    int n = buffer.ToArray().Length;
                    cArr = buffer.ToArray();
                    double test = 0;
                    for (int i = 0; i < n; i++)
                    {
                        int r = cArr[i].r;
                        int g = cArr[i].g;
                        int b = cArr[i].b;

                        double temp = 0.2126 * (double)r + 0.7152 * (double)g + 0.0722 * (double)b;
                        test = test + temp;
                    }
                    lumenValue = test;
                    Debug.Log("Lumen: " + test);


                _requests.Dequeue();
            }
            else
            {
                break;
            }
        }
    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {

        if (Time.frameCount % 90 == 0)
        {
            if (_requests.Count < 1)
            {
                _requests.Enqueue(AsyncGPUReadback.Request(source));
                Debug.Log("requests." + _requests.Count);
            }
            else
                Debug.Log("Too many requests.");
        }
       

        Graphics.Blit(source, destination);
    }


}



'use client'

import {useState} from 'react';
export default function test(){

    const [res,setRes] = useState("")
    
const test = async () => {
   
      const response = await fetch('http://127.0.0.1:8000/', {
        method: 'GET',
      });
  
  
      const data = await response.json();
      setRes(data.message)
      console.log('Response from FastAPI:', data);
   
  };
  
    return(<div onClick = {test}>
<div>FAST</div>
{res && (
    <p>{`${res}`}</p>
      )}
    </div>)
}
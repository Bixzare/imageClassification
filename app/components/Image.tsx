"use client";
import React, { useState } from "react";

export default function App() {
  const processOptions = ["Mosquito", "Cancer", "Pneumonia"]; // Example options

  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [process, setProcess] = useState(processOptions[0]); // Set default value
  const [uploaded, setUploaded] = useState(false);
  const [isLoading, setIsloading] = useState(false);
  const [res, setRes] = useState<any>("");
  const [pred, setPred] = useState("");

  const handleImageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      // Generate a preview URL for the image
      const previewUrl = URL.createObjectURL(selectedFile);

      // Create a FileReader instance to read the image as Base64
      const reader = new FileReader();
      reader.onloadend = () => {
        // Get the Base64 string from the result of FileReader
        const base64Image = reader.result as string; // Typecast to string
        // Update the state with the preview URL, file, uploaded flag, and Base64 string
        setPreview(previewUrl);
        setFile(selectedFile);
        setUploaded(true);
        setRes("");
      };

      // Read the image file as Base64
      reader.readAsDataURL(selectedFile);
    }
  };

  const submitImage = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!file) return;

    setIsloading(true);
    const formData = new FormData();
    formData.append("image", file);
    formData.append("process", process); // Adding text data to the form

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData, // No need to manually set Content-Type
      });

      if (!response.ok) throw new Error("Failed to upload image");

      const data = await response.json();
      console.log(
        `Success! Uploaded: ${data.process}, ${data.imageName}, ${data.result}`
      );
      setRes(data.result.classes);
      setPred(data.result.pred);
      console.log("result",res,"pred",pred,"class")
    } catch (error) {
      console.error("Error:", error);
    }

    setIsloading(false);
  };

  return (
    <div className=" h-auto w-auto p-8 bg-white rounded-md shadow-xl flex flex-col items-center">
      <h1 className="text-black flex justify-start text-2xl mb-2">
        Classfiication Models
      </h1>
      <form
        onSubmit={submitImage}
        className="flex flex-col self-start space-y-4"
      >
        <div className="mb-4">
          <label htmlFor="process" className="mr-2 text-black">
            Select Model
          </label>
          <select
            id="process"
            value={process}
            onChange={(e) => setProcess(e.target.value)}
            className="border p-2 rounded text-black"
          >
            {processOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </div>

        <input
          type="file"
          name="image"
          accept="image/*"
          onChange={handleImageChange}
          required
          className={`mb-2 text-black border ${
            uploaded ? "border-green-500" : "border-grey-300"
          } p-2 rounded $`}
        />

        <button
          type="submit"
          className="bg-green-600 text-white py-2 px-4 rounded hover:text-green-500 hover:bg-white transition-all durration-200 rounded border border-transparent hover:border-green-500 "

        >
          Process Image
        </button>
      </form>

      {preview && (
        <div className="p-4">
          <img
            src={preview}
            alt="Preview"
            className="mt-4 w-[224px] h-[224px] object-cover rounded"
          />
        </div>
      )}

      {isLoading && (
        <div className="flex justify-center items-center p-4">
          <div className="loader"></div>
        </div>
      )}
      {res.length > 0 && (
          <>
          <div className ="">
          <div>{res.predicted_class}</div>
          <ul>
        {res.map((item:any, index:any) => {
          // Apply green text to the class with the highest probability (pred)
          const isMaxProbability = item.class === pred;
          
          return (
            <li
              key={index}
              className={`text-xl text-gray-700 ${isMaxProbability ? 'text-green-500 font-bold' : ''}`}
            >
              {item.class}: {item.probability}%
            </li>
          );
        })}
      </ul>
          </div>
        </>
      )}
    </div>
  );
}

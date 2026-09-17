const formData = new FormData();
formData.append("input_type", "image");
formData.append("file", imageFile);  // ya text ke liye sirf text field

fetch("/generate", { method: "POST", body: formData });
import PySimpleGUI as sg
import cv2

layout = [
    [sg.Text('Scale Factor:'),
    sg.Slider(range=(1.01,3.0), default_value=1.3, resolution=0.01, expand_x=True, enable_events=True, orientation='horizontal', key='-SCALEFACTOR-')],
    [sg.Text('Minimum Neighbors:'),
    sg.Slider(range=(0,15), default_value=7, resolution=1, expand_x=True, enable_events=True, orientation='horizontal', key='-MINNEIGHBOR-')],
    [sg.Image(key='-IMAGE-')],
    [sg.Text('People in picture: 0', key='-TEXT-', expand_x=True, justification='c')]
]

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

window = sg.Window('Face Detector', layout)
scale_factor = 1.3
min_neighbor = 7

while True:
    event, values = window.read(timeout=0)
    if event == sg.WIN_CLOSED:
        break

    elif event == '-SCALEFACTOR-':
        scale_factor = values['-SCALEFACTOR-']

    elif event == '-MINNEIGHBOR-':
        min_neighbor = values['-MINNEIGHBOR-']

    # print(scale_factor, min_neighbor)

    _, frame = video.read()

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(
        gray,
        scaleFactor = scale_factor,
        minNeighbors = int(min_neighbor),
        minSize = (50,50)
    )
    print(face)

    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0) ,2)

    imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    window['-IMAGE-'].update(data=imgbytes)

    window['-TEXT-'].update(f"People in picture: {len(face)}")

window.close()

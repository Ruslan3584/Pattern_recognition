import asyncio
import websockets
import numpy as np


def parsing_numbers(f,etal_scale1,etal_scale2):
    f = np.fromstring(f, dtype=int, sep='\n')
    f = np.delete(f, 0)
    c = list(range(etal_scale1*etal_scale2,len(f),(etal_scale1*etal_scale2)+1 ))    # indexes of digits
    f = np.delete(f, c)                                                             # deleting digits
    c = list(range(0,len(f)+etal_scale1*etal_scale2 ,etal_scale1*etal_scale2))
    numbers = []
    for i in range(0,len(c)-1):
        numbers.append(f[c[i]:c[i+1]].reshape((etal_scale1,etal_scale2)))
    return numbers      # return etalone_matrixes



def parsing_task(x,g,h):
    x = np.fromstring(x, dtype=int, sep='\n')
    global STEP
    STEP = x[0]
    x = np.delete(x, 0)
    return x.reshape((g,h))




def recognition(task,noise_level, tsk_scale1, tsk_scale2, numbers):        
    result = []
    for i in numbers:
        d = []
        noise = np.bitwise_xor(i, task)            # Xoring matrixes ( task + etalone_matr) 
        
        noise = noise.reshape(1, noise.shape[0]*noise.shape[1])         # reshape noise (from matr to vector)
        
        for j in range(0, noise.shape[0]*noise.shape[1]):           
            if noise[0][j] == 0:
                d.append(1-noise_level)
            else:
                d.append(noise_level)
                '''
                log(.) is a monotonic function,
                k* = argmax(f) =>  k* = argmax(log(f))
                Used log(.) for better performance

                '''
        result.append(np.sum(np.log10(d)))              
    
    return np.argmax(result)           # returning  index of our guesses







async def hello():
    uri = "wss://sprs.herokuapp.com/first/task2"     # task2 - session id

    async with websockets.connect(uri) as websocket:
        message = "Let's start"
        await websocket.send(message)
        tsk = await websocket.recv()           # Receiving a string [width] [height] [Number_of_etalone_matrixes] from the server
        print(tsk)

        global etal_scale1,etal_scale2             # etal_scale1,etal_scale2 - shapes of etalone-matrixes
        etal_scale2, etal_scale1, Number_of_etalone_matrixes = [int(i) for i in tsk.split(" ")]           # Number_of_etalone_matrixes - number of etalone-matrixes

        settings = input("Settings:   [width] [height] [noise] [totalSteps] [shuffle] ")
        
        tsk_scale2, tsk_scale1, noise_level, totalSteps, shuffle = [str(i) for i in settings.split(" ")]          # these parameters were discribed in documentation
        tsk_scale2 = int(tsk_scale2)
        tsk_scale1 = int(tsk_scale1)
        noise_level = float(noise_level)
        totalSteps = int(totalSteps)
        
        await websocket.send(settings)
        tsk = await websocket.recv()


        numbers = parsing_numbers(bytes(tsk,"utf-8"), etal_scale1*tsk_scale1,etal_scale2*tsk_scale2)

        for i in range(0, totalSteps):
            await websocket.send('Ready')              # Send the message Ready to start completing the task, Receive a problem in the form
            task = await websocket.recv()
            x = bytes(task,'utf-8')
            task = parsing_task(x,etal_scale1*tsk_scale1,etal_scale2*tsk_scale2)
            res = str(recognition(task, noise_level, tsk_scale1, tsk_scale2, numbers))
            await websocket.send(str(STEP) + " " + res)
            print(f"< {await websocket.recv()}")               #  Receive a response in the form [step] answerj


        await websocket.send("Bye")               # Otherwise, send Bye
        print(f"< {await websocket.recv()}")

asyncio.get_event_loop().run_until_complete(hello())

import asyncio
import websockets
import numpy as np


def parsing_numbers(f,etal_scale1,etal_scale2):
    '''
    parse etalone numbers
    '''
    f = np.fromstring(f, dtype=int, sep='\n')
    f = np.delete(f, 0)
    c = list(range(etal_scale1*etal_scale2,len(f),(etal_scale1*etal_scale2)+1))
    '''
    delete digits
    '''
    f = np.delete(f, c)
    c = list(range(0,len(f)+etal_scale1*etal_scale2 ,etal_scale1*etal_scale2))
    numbers = []
    for i in range(0,len(c)-1):
        numbers.append(f[c[i]:c[i+1]].reshape((etal_scale1,etal_scale2)))
    '''
    return etalone matrixes
    '''
    return numbers      



def parsing_task(x,g,h):
    '''
    make np.array from string
    (for further calculations)
    '''
    x = np.fromstring(x, dtype=int, sep='\n')
    global STEP
    STEP = x[0]
    x = np.delete(x, 0)
    return x.reshape((g,h))




def recognition(task,noise_level, tsk_scale1, tsk_scale2, numbers):        
    result = []
    for i in numbers:
        d = []

        '''
        XOR matrixes ( task + etalone_matr) 
        '''
        noise = np.bitwise_xor(i, task)           
        
        '''
        reshape noise (from matr to vector)
        '''

        noise = noise.reshape(1, noise.shape[0]*noise.shape[1])
        
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
    '''
    return index of out guesses 
    '''
    return np.argmax(result)           







async def hello():
    '''
    task1 - session ID
    '''
    uri = "wss://sprs.herokuapp.com/first/task1"

    async with websockets.connect(uri) as websocket:
        await websocket.send("Let's start")
        '''
        start session and receive parameters from server
        '''
        tsk = await websocket.recv()           
        print(tsk)

        '''
        etal_scale1,etal_scale2 - shapes of etalone-matrixes
        '''
       
        etal_scale2, etal_scale1,n = [int(i) for i in tsk.split(" ")]

        settings = input("Settings:   [width] [height] [noise] [totalSteps] [shuffle] ")
        
        tsk_scale2, tsk_scale1, noise_level, totalSteps, shuffle = [str(i) for i in settings.split(" ")]
        '''
        make appropriate data types from string
        '''

        tsk_scale2 = int(tsk_scale2)
        tsk_scale1 = int(tsk_scale1)
        noise_level = float(noise_level)
        totalSteps = int(totalSteps)
        
        '''
        send parameters to server
        and receive etalone matrixes (in string)
        from server
        '''
        await websocket.send(settings)
        tsk = await websocket.recv()

        '''
        parse etalone matrixes 
        '''
        numbers = parsing_numbers(bytes(tsk,"utf-8"), etal_scale1*tsk_scale1,etal_scale2*tsk_scale2)

        for i in range(0, totalSteps):
            await websocket.send('Ready')
            '''
            receive task from server
            and parse task
            '''
            task = await websocket.recv()
            task = parsing_task(bytes(task,'utf-8'),etal_scale1*tsk_scale1,etal_scale2*tsk_scale2)
            '''
            main part of our prorgamm
            regognize numbers
            '''
            res = str(recognition(task, noise_level, tsk_scale1, tsk_scale2, numbers))
            await websocket.send(str(STEP) + " " + res)

            '''
            receive answers from server 
            '''
            print(f"< {await websocket.recv()}")              


        await websocket.send("Bye")
        '''
        receive result from server
        '''
        print(f"< {await websocket.recv()}")

asyncio.get_event_loop().run_until_complete(hello())

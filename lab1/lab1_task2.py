import asyncio
import websockets
import numpy as np

def parsing_heatmap(x):

    '''
    check parsing
    >>> parsing_heatmap('77 17 1 65 109 189 197 187 205 231')
    array([0.06025039, 0.01330203, 0.00078247, 0.05086072, 0.08528951,
           0.14788732, 0.1541471 , 0.14632238, 0.16040689, 0.18075117])
    >>> parsing_heatmap(['a','b','c'])
    Traceback (most recent call last):
    ...
    AttributeError: 'list' object has no attribute 'find'
    '''    




    '''
    make np.array from string and then normalize received heatmap
    '''
    heatmap = np.array(x[x.find("\n")+1:].split(" "),dtype = 'float64')
    heatmap = heatmap/np.sum(heatmap)              
    return heatmap


def search_k(heatmap):


    '''
    check function
    >>> search_k(np.array([0.1,0.4,0.3]))
    1
    >>> search_k('qqq')
    Traceback (most recent call last):
    ...
    TypeError: unsupported operand type(s) for +: 'int' and 'str'
    '''    


    '''
    metric L1 (|d-d*|)
    '''

    for i in range(1,len(heatmap)):                             
        if sum(heatmap[:i]) >= 0.5:
            return i-1


def delta_loss(heatmap, delta):
    '''
    check delta
    >>> delta_loss(np.array([0.06025039, 0.01330203, 0.00078247, 0.05086072, 0.08528951,0.14788732, 0.1541471 , 0.14632238, 0.16040689, 0.18075117]),3)
    7
    >>> delta_loss(np.array([0.06025039, 0.01330203, 0.00078247, 0.05086072, 0.08528951,0.14788732, 0.1541471 , 0.14632238, 0.16040689, 0.18075117]),100)
    Traceback (most recent call last):
    ...
    ValueError: attempt to get argmax of an empty sequence
    '''    


    result = []
    '''
    metric delta_loss (delta > 0)
    '''
    for i in range(0,len(heatmap)-delta+1):
        result.append(sum(heatmap[i:i+delta]))
    return np.argmax(result)



async def hello():
    '''
    main function for interaction with server
    '''
    uri = "wss://sprs.herokuapp.com/second/task2"
    async with websockets.connect(uri) as websocket:
        '''
        input parameters from server
        '''
        settings = input("   [width] [loss] [totalSteps] [repeats]  ")
        width,loss,totalSteps,repeats = [i for i in settings.split(" ")]
        '''
        transform parameters from string to integer
        '''
        width = int(width)
        totalSteps = int(totalSteps)
        repeats = int(repeats)
        
        '''
        send parameters to server
        '''

        await websocket.send(f"Let's start with {width} {loss} {totalSteps} {repeats}" )
        response = await websocket.recv()
        for i in range(1,totalSteps+1):
            await websocket.send("Ready")

            '''
            receive heatmap from server
            '''

            response = await websocket.recv()

            '''
            parse and normalize (!) received heatmap 
            '''

            heatmap = parsing_heatmap(response)

            '''
            our answer depends on metric (L1 or delta loss (d > 0))
            '''
            if(loss == "L1"):
                res = search_k(heatmap)
            else:
                loss = int(loss)
                res = delta_loss(heatmap,loss)
            res = str(res) + " "

            '''
            send answer repeats times 
            '''
            res = (res*repeats)[:-1]

            '''
            send res to server
            '''
            await websocket.send(str(i)+"\n" +res)
            response = await websocket.recv()
            print(response)

        await websocket.send("Bye")
        '''
        receive results from server 
        '''

        response = await websocket.recv()
        print(response)
if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(hello())
asyncio.get_event_loop().run_until_complete(hello())


if __name__ == "__main__":
    import doctest
    doctest.testmod()

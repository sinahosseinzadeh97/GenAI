import{j as r}from"./index-Dpxo-yl_.js";import"./helperFunctions-ChrP1e3B.js";import"./index-B7J2Z2jS.js";import"./svelte/svelte.js";const e="rgbdEncodePixelShader",o=`varying vec2 vUV;uniform sampler2D textureSampler;
#include<helperFunctions>
#define CUSTOM_FRAGMENT_DEFINITIONS
void main(void) 
{gl_FragColor=toRGBD(texture2D(textureSampler,vUV).rgb);}`;r.ShadersStore[e]||(r.ShadersStore[e]=o);const n={name:e,shader:o};export{n as rgbdEncodePixelShader};
//# sourceMappingURL=rgbdEncode.fragment-Be96czkD.js.map
